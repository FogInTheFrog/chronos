FROM pytorch/pytorch:latest
WORKDIR /chronos
COPY figures scripts src test pyproject.toml ./
RUN pip install -e .
RUN pip install transformers accelerate gluonts typer typer_config pyarrow datasets tensorboardX scipy
ENTRYPOINT ["python", "train.py"]
CMD ["--config", "/scripts/training/configs/custom/custom-all_datasets.yaml"]
