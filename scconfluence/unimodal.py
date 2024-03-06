import warnings
from typing import Literal

import anndata as ad
import numpy as np
import torch
from torch import nn
from torch.distributions import Poisson

from scconfluence.distributions import ZeroInflatedNegativeBinomial, NegativeBinomial
from scconfluence.base_module import BaseModule
from scconfluence.nn import VariationalEncoder, FCLayers, DecoderSCVI, DecoderPeakVI
from scconfluence.dataset_utils import format_batch, configure_unimodal_dataset


class AutoEncoder(BaseModule):
    """
    Autoencoder model for unimodal single-cell data. This is a generic model designed to
    handle different types of data (e.g. RNA, ATAC, etc.) and different types of losses
    (e.g. L2, ZINB, etc.). For high dimensional modalities, such as RNA and ATAC, the
    model can use input data different from the one used to assess the quality of the
    reconstruction. For example, the input data can be the PCA projection of the
    normalized counts and the output data can be the raw counts.

    :param adata: AnnData object containing the data.
    :param modality: string indicating the modality of the data.
    :param rep_in: string indicating the entry of the Anndata where to look for the
        input data, i.e. the data used as input of the encoder. If not None, the input
        data will be extracted from the obsm field of the AnnData object. If None, the
        input data is assumed to be  the X field of the AnnData object.
    :param rep_out: string indicating the entry of the Anndata where to look for the
        output data, i.e. the data used to compare with the output of the decoder. If
        not None, the output data will be extracted from the layers field of the AnnData
        object. If None, the output data is assumed to be  the X field of the AnnData
        object.
    :param batch_key: If the data is not composed of multiple experimental batches than
        this should be set to None. Otherwise, this string indicates the entry in the
        obs field of the Anndata where to look for the batch information.
    :param n_hidden: number of hidden units in the encoder and decoder.
    :param n_latent: number of latent dimensions.
    :param type_loss: string indicating the type of loss to use. It can be "l2", "zinb",
        "nb", "poisson" or "binary".
    :param reconstruction_weight: weight of the reconstruction loss.
    :param avg_feat: if True, the reconstruction loss is averaged over the features,
        otherwise it is summed.
    :param n_layers_enc: number of layers in the encoder.
    :param n_layers_dec: number of layers in the decoder.
    :param use_batch_norm_enc: if not None, it indicates the type of batch normalization
        to use in the encoder.
    :param use_batch_norm_dec: if not None, it indicates the type of batch normalization
        to use in the decoder.
    :param dropout_rate: dropout rate to use in the encoder and decoder.
    :param var_eps: small positive value to add to the variance of the posterior
        distribution.
    :param deeply_inject_covariates_enc: if True, the batch_index is deeply injected in
        the encoder.
    :param deeply_inject_covariates_dec: if True, the batch_index is deeply injected in
        the decoder.
    :param positive_out: if True, the output of the decoder is forced to be positive.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        modality: str,
        rep_in: str | None = None,
        rep_out: str | None = None,
        batch_key: str | None = None,
        n_hidden: int = 64,
        n_latent: int = 16,
        type_loss: Literal["l2", "zinb", "nb", "poisson", "binary"] = "l2",
        reconstruction_weight: float = 1.0,
        avg_feat: bool = True,
        n_layers_enc: int = 3,
        n_layers_dec: int = 2,
        use_batch_norm_enc: Literal[None, "ds", "standard"] = None,
        use_batch_norm_dec: Literal[None, "ds", "standard"] = None,
        dropout_rate: float = 0.0,
        var_eps: float = 1e-4,
        deeply_inject_covariates_enc: bool = True,
        deeply_inject_covariates_dec: bool = True,
        positive_out: bool = False,
    ):
        super().__init__()

        self.modality = modality
        self.dataset, dim_in, dim_out, batch_dic = configure_unimodal_dataset(
            adata=adata,
            rep_in=rep_in,
            rep_out=rep_out,
            batch_key=batch_key,
            modality=modality,
        )
        if n_hidden < n_latent:
            warnings.warn("n_hidden should be larger than n_latent")
        if n_latent > dim_in:
            raise warnings.warn(
                "n_latent should be smaller than the number of features used as input "
                "for the autoencoder"
            )
        if n_latent > dim_out:
            raise warnings.warn(
                "n_latent should be smaller than the dimension of the reconstructed "
                "data"
            )

        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.type_loss = type_loss
        self.reconstruction_weight = reconstruction_weight
        if avg_feat:
            self.feature_loss_reduction = torch.mean
        else:
            self.feature_loss_reduction = torch.sum

        self.batch_dic = batch_dic
        n_batch = len(batch_dic)
        self.n_batch = n_batch

        self.z_encoder = VariationalEncoder(
            n_layers=n_layers_enc,
            n_input=dim_in,
            n_hidden=self.n_hidden,
            n_output=self.n_latent,
            n_batch=n_batch,
            use_batch_norm=use_batch_norm_enc,
            last_activation=True,
            dropout_rate=dropout_rate,
            var_eps=var_eps,
            deeply_inject_covariates=deeply_inject_covariates_enc,
        )

        if type_loss in ["zinb", "nb", "poisson"]:
            self.decoder = DecoderSCVI(
                n_layers=n_layers_dec,
                n_input=self.n_latent,
                n_hidden=self.n_hidden,
                n_output=dim_out,
                n_batch=n_batch,
                use_batch_norm=use_batch_norm_dec,
                deeply_inject_covariates=deeply_inject_covariates_dec,
                use_poisson=type_loss == "poisson",
            )
        elif type_loss == "binary":
            self.decoder = DecoderPeakVI(
                n_layers=n_layers_dec,
                n_input=self.n_latent,
                n_hidden=self.n_hidden,
                n_output=dim_out,
                n_batch=n_batch,
                use_batch_norm=use_batch_norm_dec,
                deeply_inject_covariates=deeply_inject_covariates_dec,
            )
            self.region_factors = torch.nn.Parameter(torch.zeros(dim_out))
            self.depth_encoder = DecoderPeakVI(
                n_input=dim_out,
                n_output=1,
                n_hidden=self.n_hidden,
                n_layers=n_layers_enc,
                n_batch=n_batch,
                deeply_inject_covariates=deeply_inject_covariates_dec,
            )
        elif type_loss == "l2":
            self.decoder = FCLayers(
                n_layers=n_layers_dec,
                n_input=self.n_latent,
                n_hidden=self.n_hidden,
                n_output=dim_out,
                n_batch=n_batch,
                use_batch_norm=use_batch_norm_dec,
                last_activation=False,
                positive_out=positive_out,
                deeply_inject_covariates=deeply_inject_covariates_dec,
            )
        else:
            raise ValueError("type_loss <{}> not recognized".format(type_loss))

    def inference(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Encoding step of the model which consists in a forward pass through the model's
        encoder.

        :param x: input mini-batch of data.
        :return: the latent embeddings (parameters of the posterior distribution), the
            library size and the batch indices.
        """
        z_m, z_v, z = self.z_encoder(x["input"], x["batch_index"])
        if self.type_loss == "binary":
            library = self.depth_encoder(x["output"], x["batch_index"])
        else:
            library = torch.log(x["output"].sum(1)).unsqueeze(1)
        return dict(
            z=z, qz_m=z_m, qz_v=z_v, library=library, batch_index=x["batch_index"]
        )

    def generative(
        self, inference_output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Generative step of the model which consists in a forward pass through
        the model's decoder.

        :param inference_output: output of the encoder (inference step).
        :return: the output of the decoder.
        """
        if self.type_loss in ["zinb", "nb"]:
            x_scale, x_var, x_r, x_dropout = self.decoder(
                dispersion="gene-cell",
                z=inference_output["z"],
                log_library=inference_output["library"],
                batch_index=inference_output["batch_index"],
            )
            # clamp values for stability
            x_var = torch.exp(torch.clamp(x_var, min=-15, max=15))
            return dict(x_scale=x_scale, x_r=x_r, x_var=x_var, x_dropout=x_dropout)
        elif self.type_loss == "poisson":
            x_r = self.decoder(
                dispersion="",
                z=inference_output["z"],
                log_library=inference_output["library"],
                batch_index=inference_output["batch_index"],
            )[2]
            return dict(x_r=x_r)
        else:
            x_r = self.decoder(inference_output["z"], inference_output["batch_index"])
            return dict(x_r=x_r)

    def reconstr_loss(
        self,
        x: dict[str, torch.Tensor],
        inference_output: dict[str, torch.Tensor],
        generative_output: dict[str, torch.Tensor],
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute the reconstruction loss.

        :param x: input mini-batch of data.
        :param inference_output: output of the encoder (inference step).
        :param generative_output: output of the decoder (generative step).
        :param reduce: whether to reduce the cell reconstruction losses to a single
            scalar or not.
        :return: the reconstruction loss.
        """
        loss = torch.tensor([[0.0]])
        if self.type_loss == "l2":
            loss = nn.MSELoss(reduction="none")(x["output"], generative_output["x_r"])
        elif self.type_loss == "zinb":
            mu = generative_output["x_r"]
            theta = generative_output["x_var"]
            zi_logits = generative_output["x_dropout"]
            loss = -ZeroInflatedNegativeBinomial(
                mu=mu, theta=theta, zi_logits=zi_logits
            ).log_prob(x["output"])
        elif self.type_loss == "nb":
            mu = generative_output["x_r"]
            theta = generative_output["x_var"]
            loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x["output"])
        elif self.type_loss == "poisson":
            loss = -Poisson(generative_output["x_r"]).log_prob(x["output"])
        elif self.type_loss == "binary":
            region_p = (
                torch.sigmoid(self.region_factors)
                if self.region_factors is not None
                else 1
            )
            p_r = generative_output["x_r"] * region_p * inference_output["library"]
            loss = self.feature_loss_reduction(
                nn.BCELoss(reduction="none")(p_r, (x["output"] > 0).float()), dim=-1
            )

        loss = self.feature_loss_reduction(loss, dim=-1)
        if reduce:
            loss = loss.mean()
        return loss

    def loss(
        self,
        x: dict[str, torch.Tensor],
        inference_output: dict[str, torch.Tensor],
        generative_output: dict[str, torch.Tensor],
        reduce: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss terms of the model. For an AutoEncoder model, this means only
        the reconstruction loss but for future developments other losses can be added to
        this model.

        :param x: mini-batch input data.
        :param inference_output: output of the encoder (inference step).
        :param generative_output: output of the decoder (generative step).
        :param reduce: whether to reduce the cell reconstruction losses to a single
            scalar or not.
        :return: a dictionary containing all loss terms used to train the model.
        """
        r_loss = self.reconstr_loss(
            x=x,
            inference_output=inference_output,
            generative_output=generative_output,
            reduce=reduce,
        )
        self.metrics_to_log.append(
            {
                "name": "{}_{}".format(self.modality, "reconstruction_loss"),
                "value": r_loss,
                "batch_size": x["input"].size(0),
                "reduce_fx": torch.mean,
            }
        )
        return {"reconstruction_loss": self.reconstruction_weight * r_loss}

    def latent_batch(self, x: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """
        Get latent embeddings for a mini-batch of data. Used for prediction after the
        training of the model.

        :param x: mini-batch of input data.
        :return: latent embeddings and their corresponding observation names.
        """
        inf_dic = self.inference(x)
        return inf_dic["qz_m"].detach().cpu().numpy(), np.array(x["cell_index"])

    def predict_step(self, batch, batch_idx):
        """
        Predict the latent embeddings for a mini-batch of data.

        :return: latent embeddings and their corresponding observation names.
        """
        batch = format_batch(batch)
        latents, cell_idxes = self.latent_batch(batch)
        return {"latents": latents, "cell_idxes": cell_idxes}

    def log_latent_norms(
        self, inference_dic: dict[str, torch.Tensor], return_log: bool = False
    ):
        """
        Log the norm of the latent embeddings.

        :param inference_dic: output of the encoder (inference step).
        :param return_log: whether to return the log or not.
        :return: if return_log is true, return the log to be printed.
        """
        log = {
            "name": f"{self.modality}_z_norms",
            "value": torch.linalg.norm(inference_dic["z"], dim=1).mean(),
            "batch_size": inference_dic["z"].size(0),
            "reduce_fx": torch.mean,
        }
        self.metrics_to_log.append(log)
        if return_log:
            return log

    def reset_log_metrics(self):
        """
        Reset the list of metrics to log.
        """
        self.metrics_to_log = []
