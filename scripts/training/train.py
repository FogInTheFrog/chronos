# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import sys
import json
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

import typer
from scipy.stats import norm
from transformers.modeling_outputs import Seq2SeqLMOutput
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import torch.nn.functional as F
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    Trainer,
    TrainingArguments,
)
import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

from chronos import ChronosConfig, ChronosTokenizer

import torch.nn as nn
import torch.special as special

from transformers import T5ForConditionalGeneration


class T5EnergyHeadWrapper(T5ForConditionalGeneration):
    """A wrapper that takes a T5 and adds an extra layer on its output. This
    layer takes logit output h (of T5) and transforms it into a quadratic
    energy functional on the space of tokens."""

    def __init__(self, config, forecast_points):
        super().__init__(config)

        self.n_special_tokens = config.use_eos_token + 1
        self.n_regular_tokens = config.n_tokens - self.n_special_tokens
        assert len(forecast_points) == self.n_regular_tokens
        self.register_buffer('forecast_points', forecast_points)

        self.quad_energy_head = nn.Linear(config.n_regular_tokens, config.n_tokens)
        self.lin_energy_head = nn.Linear(config.n_regular_tokens, config.n_tokens)
        self.bias_energy_head = nn.Linear(config.n_tokens, config.n_tokens)

        torch.nn.init.xavier_uniform_(self.quad_energy_head.weight)
        torch.nn.init.xavier_uniform_(self.lin_energy_head.weight)
        torch.nn.init.xavier_uniform_(self.bias_energy_head.weight)


    def forward(self, labels=None, return_dict=None, **kwargs):

        t5_output = super().forward(
            labels=labels, return_dict=return_dict, **kwargs
        )
        t5_logits = t5_output.logits
        t5_regular_logits = t5_logits[self.n_special_tokens:]

        quad_energy_coefs = self.quad_energy_head(t5_regular_logits)
        lin_energy_coefs = self.lin_energy_head(t5_regular_logits)
        bias_energy_coefs = self.bias_energy_head(t5_logits)
        special_bias_energy_coefs = bias_energy_coefs[:self.n_special_tokens]
        regular_bias_energy_coefs = bias_energy_coefs[self.n_special_tokens:]

        # Compute the energy functional.
        special_energy = special_bias_energy_coefs
        regular_energy \
            = quad_energy_coefs * (self.forecast_points ** 2) \
            + lin_energy_coefs * self.forecast_points \
            + regular_bias_energy_coefs
        energy = torch.cat([special_energy, regular_energy])

        # Compute the Boltzmann distribution associated with this energy.
        probs = F.softmax(-energy)
        log_probs = F.log_softmax(-energy)

        loss_fn = nn.NLLLoss(ignore_index=-100, reduction="mean")
        loss = loss_fn(log_probs, labels.view(-1))
        print(f"{loss.item()=}")

        probs = probs.view(t5_logits.shape[0], t5_logits.shape[1], probs.shape[1])
        if not return_dict:
            return loss, probs

        t5_output["loss"] = loss
        t5_output["logits"] = probs
        return t5_output


app = typer.Typer(pretty_exceptions_enable=False)


def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> Dict:
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["transformers_version"] = transformers.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(ckpt_path: Path, training_config: Dict):
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
        )


def get_next_path(
        base_fname: str,
        base_dir: Path,
        file_type: str = "yaml",
        separator: str = "-",
):
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + 1
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


def load_model(
        model_id="google/t5-efficient-tiny",
        model_type="seq2seq",
        vocab_size=4096,
        random_init=False,
        tie_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        boundaries=None,
):
    """
    Load the specified HuggingFace model, adjusting the vocabulary
    size, special token IDs, and initialization options.

    This allows to set a model up for training on a new vocabulary
    of tokens.
    """
    assert model_type in ["seq2seq", "causal"]
    AutoModelClass = (
        T5EnergyHeadWrapper if model_type == "seq2seq" else None
    )  # Load T5ForMeanScale for seq2seq model type

    print(f"boundaries: {boundaries}")

    if random_init:
        print("Using random initialization")
        config = T5Config.from_pretrained(model_id)

        # Modify config as needed for T5ForMeanScale
        config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings

        model = AutoModelClass(config, forecast_points=boundaries)
    else:
        print(f"Using pretrained initialization from {model_id}")
        model = AutoModelClass.from_pretrained(model_id)

    model.resize_token_embeddings(vocab_size)

    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id

    return model


def has_enough_observations(
        entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
            len(entry["target"]) >= min_length
            and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"debug"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
            self,
            datasets: list,
            probabilities: List[float],
            tokenizer: ChronosTokenizer,
            context_length: int = 512,
            prediction_length: int = 64,
            drop_prob: float = 0.2,
            min_past: Optional[int] = None,
            model_type: str = "seq2seq",
            imputation_method: Optional[MissingValueImputation] = None,
            mode: str = "training",
            np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "debug")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            # Causal models do not play nice with missing values, so it is
            # recommended to use an imputation method, e.g., LastValueImputation
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "debug", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "debug": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("debug").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "debug":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)


@app.command()
@use_yaml_config(param_name="config")
def main(
        training_data_paths: str,
        probability: Optional[str] = None,
        context_length: int = 512,
        prediction_length: int = 64,
        min_past: int = 64,
        max_steps: int = 200_000,
        save_steps: int = 50_000,
        log_steps: int = 500,
        per_device_train_batch_size: int = 32,
        learning_rate: float = 1e-3,
        optim: str = "adamw_torch_fused",
        shuffle_buffer_length: int = 100,
        gradient_accumulation_steps: int = 2,
        model_id: str = "google/t5-efficient-tiny",
        model_type: str = "seq2seq",
        random_init: bool = False,
        tie_embeddings: bool = False,
        output_dir: str = "./output/",
        tf32: bool = True,
        torch_compile: bool = True,
        tokenizer_class: str = "MeanScaleUniformBins",
        tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
        n_tokens: int = 4096,
        n_special_tokens: int = 2,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        use_eos_token: bool = True,
        lr_scheduler_type: str = "linear",
        warmup_ratio: float = 0.0,
        dataloader_num_workers: int = 1,
        max_missing_prop: float = 0.9,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: Optional[int] = None,
):
    if tf32 and not (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2 ** 32)

    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    assert isinstance(training_data_paths, list)

    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    assert isinstance(probability, list)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    assert model_type in ["seq2seq", "causal"]

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets "
        f"for training: {training_data_paths}",
        logger,
    )

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    shuffled_train_dataset_old = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type=model_type,
        imputation_method=LastValueImputation() if model_type == "causal" else None,
        mode="training",
    )

    shuffled_train_dataset = shuffled_train_dataset_old.shuffle(shuffle_buffer_length=shuffle_buffer_length)

    log_on_main("Initializing model", logger)

    model = load_model(
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        boundaries=shuffled_train_dataset_old.tokenizer.boundaries
    )

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__

    # Define training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):

            # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3328
            # as in standard compute_loss
            # super().compute_loss(self, model, inputs)
            # if self.label_smoother is not None and "labels" in inputs:
            #     labels = inputs.pop("labels")
            # else:
            #     labels = None
            outputs = model(**inputs)

            if isinstance(outputs, dict):

                loss = outputs["loss"]
                logits = outputs["logits"]
            else:
                loss, logits = outputs[0], outputs[1]

            return (loss, logits) if return_outputs else loss

    # original_boundaries = deepcopy(shuffled_train_dataset_old.tokenizer.boundaries)
    # Create Trainer instance
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
    )
    log_on_main("Training", logger)

    trainer.train()

    trained_boundaries = model.boundaries
    # print(original_boundaries)
    # print(trained_boundaries)
    # Calculate the difference
    # difference = original_boundaries - trained_boundaries

    # Sum the elements of the difference
    # sum_difference = torch.sum(difference)

    # Print the sum of the difference
    # print("Sum of the differences between tensors:", sum_difference.item())

    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final", training_config=raw_training_config
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler('app_entire_training.log', maxBytes=1024 * 1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Add file handler to logger
    logger.addHandler(file_handler)
    app()