import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from collections import defaultdict
from typing import Union, Optional


# TODO: implement a mechanism to clean up local disk space by deleting old runs in the wandb directory (? - dangerous)


class WandbLogger:
    """
    Handles logging of metrics, images, model checkpoints
    and datasets to Weights and Biases.
    """

    def __init__(
        self,
        project: str,
        entity: str,
    ) -> None:
        self.project = project
        self.entity = entity
        self.metrics = defaultdict(list)

    def init_run(self, model, is_sweep: bool = False):
        if is_sweep:
            assert (
                wandb.run is not None
            ), "for sweeps, you should call wandb.init() before initializing the logger."
            self.run = wandb.run
        else:
            self.run = wandb.init(
                project=self.project, entity=self.entity, save_code=True, mode="online"
            )
        self.run.watch(model, log="all")

    def log_metric(self, value, name: str, step: Optional[int] = None):
        self.metrics[name].append(value)
        self.run.log({name: value}, step=step)

    def log_tensor_as_image(
        self, images: Union[list, torch.Tensor], name: str, step: int
    ):
        """
        :param images: tensor of shape (B, C, H, W) or list
        of tensors of common shape (C, H, W)
        """
        pil_image = self.tensor_to_PIL_image(images)
        self.run.log({name: wandb.Image(pil_image)}, step=step)

    def log_plot(self, name: str, step: int):
        self.run.log({name: wandb.Image(plt)}, step=step)

    def log_table(self, df: pd.DataFrame, name: str, step: int):
        """
        There is an issue with the visualization: key "0.01" is displayed as "0\.01".
        :param df: pandas DataFrame
        :param name: name of the table
        :param step: current training step
        """
        wandb_table = wandb.Table(dataframe=df)
        self.run.log({name: wandb_table}, step=step)

    def log_checkpoint(self, chkpt_dir: str, artifact_name: str):
        model_artifact = wandb.Artifact(artifact_name, type="model")
        model_artifact.add_dir(chkpt_dir)
        self.run.log_artifact(model_artifact)

    @staticmethod
    def log_dataset(dataset, metadata: dict, project: str, entity: str):
        """
        Log a dataset as an artifact to Weights and Biases.
        Meant to be called once, after the dataset has been saved to disk.
        :param dataset: dataset to log
        :param metadata: dictionary with metadata about the dataset.
        must contain the following keys: id, dataset_dir.
        :param project: name of the project
        :param entity: name of the entity
        """
        notes = f"Upload dataset {metadata['id']} as an artifact."
        with wandb.init(project=project, entity=entity, notes=notes) as run:
            dataset_artifact = wandb.Artifact(
                name=metadata["id"], type="dataset", metadata=metadata
            )
            dataset_artifact.add_dir(metadata["dataset_dir"])
            run.log_artifact(dataset_artifact)

    def use_dataset(self, metadata: dict):
        """
        Log to wandb that the dataset has been used for training/testing.
        :param metadata: dictionary with metadata about the dataset.
                         must contain the following keys: id, use_as.
        """
        try:
            self.run.use_artifact(f"{metadata['id']}:latest", use_as=metadata["use_as"])
        except wandb.errors.CommError as e:
            # not sure if this exception is too broad
            print(
                f"Tried to log usage of dataset artifact {metadata['id']}, but it was not found."
            )
            if "mnist" in metadata["id"] or "cifar" in metadata["id"]:
                print(
                    "Continuing without logging the dataset usage, as it is a common dataset."
                )
            else:
                raise e

    def add_to_config(self, hyperparameters: dict, prefix: str = ""):
        if prefix:
            hyperparameters = {f"{prefix}/{k}": v for k, v in hyperparameters.items()}
        self.run.config.update(hyperparameters)

    def end_run(self):
        self.run.finish()

    @staticmethod
    @torch.no_grad()
    def tensor_to_PIL_image(images: Union[list, torch.Tensor]) -> Image.Image:
        """
        Convert a tensor to a PIL image using `torchvision.utils.make_grid`
        without normalizing the image.
        :param images: Tensor of shape (B, C, H, W) or list of tensors
        of common shape (C, H, W). expected values in range [-1, 1].
        :return: PIL image ready to be logged
        """
        # If it's a single image, it does nothing; otherwise, it stitches the images together
        grid = torchvision.utils.make_grid(images, normalize=False, nrow=4)
        # go to the [0, 1] range
        grid = (grid + 1) / 2
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        image = Image.fromarray(ndarr)
        return image
