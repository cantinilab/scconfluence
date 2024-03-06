from abc import abstractmethod
import warnings
import os
from typing import Literal
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader

from scconfluence.dataset_utils import format_batch, BatchSampler, inference_dl_trainer


class BaseModule(pl.LightningModule):
    """
    Base class for all models. It is a subclass of pytorch_lightning.LightningModule.
    """

    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()

        self.trained = False

        self.lr = None
        self.predict_mode = "latent"
        self.metrics_to_log = []

    @abstractmethod
    def inference(self, x):
        pass

    @abstractmethod
    def generative(self, inference_output):
        pass

    @abstractmethod
    def loss(self, x, inference_output, generative_output, reduce=True):
        pass

    @abstractmethod
    def log_latent_norms(self, inference_dic, return_log=False):
        pass

    @abstractmethod
    def reset_log_metrics(self):
        pass

    @abstractmethod
    def latent_batch(self, batch):
        pass

    def forward(self, x, return_loss=False):
        """
        Forward pass of the model. It returns the reconstruction and the latent
        embeddings of the input data and optionally the full loss.

        :param x: input mini-batch
        :param return_loss: whether to return the full loss or not.
        """
        inference_dic = self.inference(x)
        reconstr_dic = self.generative(inference_output=inference_dic)

        if return_loss:
            losses = self.loss(
                x=x,
                inference_output=inference_dic,
                generative_output=reconstr_dic,
                reduce=True,
            )
            full_loss = sum(losses.values())
            self.metrics_to_log.append(
                {
                    "name": "full_loss",
                    "value": full_loss,
                    "batch_size": 1,
                    "reduce_fx": torch.mean,
                }
            )
            return reconstr_dic, inference_dic, full_loss
        return reconstr_dic, inference_dic

    def fit(
        self,
        save_path: str | Path,
        use_cuda: bool = True,
        ratio_val: float = 0.2,
        batch_size: int = 512,
        pin_memory: bool = True,
        num_workers: int = 0,
        max_epochs: int = 1000,
        lr: float = 0.003,
        early_stopping: bool = True,
        es_metric: str = "val_full_loss",
        patience: int = 40,
        es_mode: Literal["min", "max"] = "min",
        test_mode: bool = False,
        save_models: bool = False,
        **trainer_kwargs,
    ):
        """
        Train the model on the dataset witch which it was initialized.

        :param save_path: path in which logs and model checkpoints will be saved.
        :param use_cuda: whether to use GPU acceleration if cuda is available.
        :param ratio_val: ratio of the dataset to be used for the validation split.
        :param batch_size: size of the mini-batches used for training. Not to be
            confused with the experimental batches.
        :param pin_memory: If True, the data loader will copy Tensors into device/CUDA
            pinned memory before returning them.
        :param num_workers: how many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
        :param max_epochs: max number of epochs used for training. Convergence is often
            reached before this number.
        :param lr: learning rate,used for training.
        :param early_stopping: whether to use early stopping.
        :param es_metric: which logged metric to use for early stopping.
        :param patience: Number of epochs to wait before stopping training if the
            monitored early stopping metric does not improve.
        :param es_mode: "min" or "max depending on whether the early stopping metric
            should be minimized or maximized.
        :param test_mode: when testing, the model will only be trained for 5 epochs with
            only 10 mini-batches per epoch.
        :param save_models: whether to save the checkpoint of the best model according
            to the early stopping metric.
        :param trainer_kwargs: additional arguments to be passed to the
            pytorch_lightning.Trainer.
        """
        indices_train, indices_val = self.dataset.split_train_val(ratio_val=ratio_val)
        tr_loader = DataLoader(
            self.dataset,
            sampler=BatchSampler(
                indices=indices_train,
                shuffle=True,
                batch_size=batch_size,
                drop_last=False,
            ),
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            self.dataset,
            sampler=BatchSampler(
                indices=indices_val,
                shuffle=True,
                batch_size=batch_size,
                drop_last=False,
            ),
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        self.lr = lr

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_path)
        profiler = SimpleProfiler(dirpath=save_path, filename="profiler")
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path, save_top_k=1, monitor=es_metric, save_weights_only=False
        )
        es_callback = EarlyStopping(monitor=es_metric, mode=es_mode, patience=patience)
        callbacks = []
        if early_stopping:
            callbacks = [es_callback, checkpoint_callback]

        device = "cpu"
        if use_cuda:
            if torch.cuda.is_available():
                device = "gpu"
            else:
                warnings.warn(
                    "use_cuda has been set to True but cuda is not available on this "
                    "machine."
                )

        fast_dev_run = False
        if test_mode:
            fast_dev_run = 10
            max_epochs = 5

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=device,
            devices=1,
            logger=tb_logger,
            default_root_dir=save_path,
            callbacks=callbacks,
            enable_checkpointing=True,
            fast_dev_run=fast_dev_run,
            profiler=profiler,
            **trainer_kwargs,
        )
        trainer.fit(model=self, train_dataloaders=tr_loader, val_dataloaders=val_loader)
        self.trained = True
        if early_stopping and not test_mode:
            self.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"]
            )
            if not save_models:
                if os.path.exists(checkpoint_callback.best_model_path):
                    os.remove(checkpoint_callback.best_model_path)

    def get_latent(
        self,
        use_cuda: bool = True,
        batch_size: int = 512,
        pin_memory: bool = True,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        """
        Get the latent embeddings of all cells of the dataset from the trained model.

        :param use_cuda: whether to use GPU acceleration if available.
        :param batch_size: size of the mini-batches used for training. Not to be
            confused with the experimental batches.
        :param pin_memory: If True, the data loader will copy Tensors into device/CUDA
            pinned memory before returning them.
        :param num_workers: how many subprocesses to use for data loading. 0 means that
            the data will be loaded in the main process.
        :return: dataframe with the latent embeddings of all cells of the dataset,
            indexed by their observation names.
        """
        if not self.trained:
            raise ValueError(
                "Model has not been trained yet. Use .fit() to train the model before "
                "imputing. If you want the imputations nonetheless, set the attribute "
                "<trained> to true manually for the  model and rerun this function.\n"
            )
        loader, trainer = inference_dl_trainer(
            self.dataset,
            use_cuda=use_cuda,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        self.predict_mode = "latent"
        predict_results = trainer.predict(model=self, dataloaders=loader)

        # format results
        latents = pd.DataFrame(
            np.concatenate(
                [batch_res_dic["latents"] for batch_res_dic in predict_results]
            ),
            index=np.concatenate(
                [batch_res_dic["cell_idxes"] for batch_res_dic in predict_results]
            ),
        )
        return latents

    def train_val_step(self, batch, split):
        """
        Wrapper to perform a training or validation step. It returns the loss of the
        mini-batch.

        :param batch: input data mini-batch
        :param split: whether the mini-batch is part of the training or validation set.
        :return: loss of the model on the mini-batch.
        """
        self.reset_log_metrics()
        batch = format_batch(batch)
        _, inference_dic, loss = self.forward(batch, return_loss=True)
        self.log_latent_norms(inference_dic)
        self.log_all_metrics(split)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Use the `train_val_step` method to perform a training step.

        :return: loss of the model on the mini-batch.
        """
        return self.train_val_step(batch=batch, split="train")

    def validation_step(self, batch, batch_idx):
        """
        Use the `train_val_step` method to perform a validation step.

        :return: loss of the model on the mini-batch.
        """
        return self.train_val_step(batch=batch, split="validation")

    def log_all_metrics(self, split: Literal["train", "val"]):
        """
        Log all metrics in self.metrics_to_log for the given split.

        :param split: whether the metrics are being logged for a training or validation
            mini-batch.
        """
        if split == "train":
            prefix = "tr"
            base_log_params = {"on_step": True, "on_epoch": False}
        else:
            prefix = "val"
            base_log_params = {"on_step": False, "on_epoch": True}
        for metric_dic in self.metrics_to_log:
            log_params = base_log_params.copy()
            log_params.update(metric_dic)
            log_params["name"] = "{}_{}".format(prefix, metric_dic["name"])
            self.log(**log_params)

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer used for training.

        :return: the optimizer object.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """
        Resets the gradients of the optimized tensors. We set the gradients to None to
        improve the performance.
        """
        optimizer.zero_grad(set_to_none=True)
