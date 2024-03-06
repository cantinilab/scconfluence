import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mudata as md

from geomloss import SamplesLoss
import ot

from scconfluence.base_module import BaseModule
from scconfluence.unimodal import AutoEncoder
from scconfluence.dataset_utils import (
    format_batch,
    inference_dl_trainer,
    configure_multimodal_dataset,
)


def check_aes(
    mdata: md.MuData,
    unimodal_aes: dict[str, AutoEncoder],
):
    """
    Check that the AutoEncoders and the mdata object are consistent, raises a ValueError
    if not.

    :param mdata: the input data
    :param unimodal_aes: dictionary of AutoEncoder objects for each modality
    """
    latent_dims = [unimodal_aes[mod].n_latent for mod in unimodal_aes]
    error_message = ""
    error_flag = False
    if len(set(latent_dims)) > 1:
        error_message += (
            "All modality AutoEncoders should have the same latent dimension.\n"
        )
    if set(mdata.mod.keys()) != set(unimodal_aes.keys()):
        error_message += "Modality names in mdata and unimodal_AEs should match.\n"
        if len(set(mdata.mod.keys()) - set(unimodal_aes.keys())) > 0:
            error_message += (
                f"The following modalities are missing from unimodal_AEs: "
                f"{set(mdata.mod.keys()) - set(unimodal_aes.keys())}\n"
            )
            if len(set(mdata.mod.keys()) - set(unimodal_aes.keys())) > 0:
                error_message += (
                    f"The following modalities are missing from unimodal_AEs: "
                    f"{set(unimodal_aes.keys()) - set(mdata.mod.keys())}\n"
                )
    if error_flag:
        raise ValueError(error_message)


class ScConfluence(BaseModule):
    """
    The ScConfluence model aims to learn a common latent space for multiple data
    modalities from unpaired measurements.

    :param mdata: the input data
    :param unimodal_aes: dictionary of AutoEncoder objects for each modality
    :param mass: mass parameter for the IOT loss which corresponds to the proportion of
        the cells (between 0 and 1) that will be matched across modalities in each
        mini-batch during training, decreasing this parameter will lead to more
        robustness towards modality-specific cell populations which should not be
        aligned with cells from other modalities (even for datasets where the same cell
        populations are roughly expected in the different modalities 0.5 is a good
        default as transporting less mass is less problematic than transporting too
        much).
    :param reach: reach parameter which controls the unbalancedness of the Sinkhorn
        regularization. Lower values will lead to a more unbalanced sinkhorn divergence.
        When trying to enforce a more complete mixing of cells from different modalities
        in the latent space, a higher reach value can be used. Only values between 0.1
        and 5. are recommended.
    :param blur: blur parameter for the Sinkhorn regularization. This parameter controls
        the strength of the entropic term in the sinkhorn regularization. Increasing
        this parameter makes the computation of the sinkhorn term faster but less
        accurate.
    :param iot_loss_weight: weight of the IOT loss in the final loss. It can be set
        higher than its default value of 0.01 in situations where the cost matrix
        between modalities is assumed to be of very high quality (e.g. NOT when
        comparing scRNA expressions and scATAC-derived gene activities). Only values
        between 0.005 and 0.1 are recommended.
    :param sinkhorn_loss_weight: weight of the Sinkhorn regularization term in the final
        loss. Setting it higher than its default value of 0.1 can be useful when trying
        to enforce a more complete mixing of cells from different modalities. On the
        opposite, when integrating more than two modalities, it can be useful to set it
        lower (e.g. to 0.3) since less regularization will be required to align the
        modalities (3 pair-wise terms are enforced). Only values between 0.01 and 0.5
        are recommended.
    """

    def __init__(
        self,
        mdata: md.MuData,
        unimodal_aes: dict[str, AutoEncoder],
        mass: float = 0.5,
        reach: float = 0.3,
        blur: float = 0.01,
        iot_loss_weight: float = 0.01,
        sinkhorn_loss_weight: float = 0.1,
    ):
        super().__init__()
        check_aes(mdata=mdata, unimodal_aes=unimodal_aes)

        self.modality_pairs = [
            name.split("cross_")[-1] for name in mdata.uns["cross_keys"]
        ]
        self.modalities = list(mdata.mod.keys())
        self.aes = nn.ModuleDict(unimodal_aes)
        self.dataset = configure_multimodal_dataset(
            mdata=mdata,
            modality_pairs=self.modality_pairs,
            unimodal_datasets={mod: ae.dataset for mod, ae in self.aes.items()},
        )

        self.mass = mass
        self.reach = reach
        self.blur = blur
        self.iot_loss_weight = iot_loss_weight
        self.sinkhorn_loss_weight = sinkhorn_loss_weight

        # inference mode attributes
        self.predict_mode = None
        self.impute_from = None
        self.impute_to = None
        self.to_batch = None

    def inference(
        self, x: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get encodings of the input data. The results of the embedding on each modality's
        cells with its corresponding AutoEncoder's encoder are returned in a dictionary
        with the same structure as the input data.

        :param x: input data mini_batch. A dictionary of dictionaries, where the first
            level correspond to the modalities and the second level correspond to the
            data for each modality.
        :return: inference results (latent embeddings, ...) for each modality
        """
        inference_output = {
            mod: self.aes[mod].inference(x[mod]) for mod in self.modalities
        }
        return inference_output

    def generative(
        self, inference_output: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Perform generative step on the inference results. The results of the generative
        step on each modality done by its corresponding AutoEncoder's decoder are
        returned in a dictionary with the same structure as the inference_output.

        :param inference_output: inference results on mini_batch. A dictionary of
            dictionaries, where the first level correspond to the modalities and the
            second level correspond to the data for each modality.
        :return: generative results for each modality, contains the reconstructed data
            (decodings of the embeddings).
        """
        generative_output = {
            mod: self.aes[mod].generative(inference_output[mod])
            for mod in self.modalities
        }
        return generative_output

    def latent_batch(
        self, x: dict[str, dict[str, torch.Tensor]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the latent embeddings for the input data. Used for prediction after the
        training of the model.

        :param x: input data mini_batch. A dictionary of dictionaries, where the first
            level correspond to the modalities and the second level correspond to the
            data for each modality.
        :return: The latent embeddings and their corresponding observation names to
            ensure proper indexing of the results.
        """
        latents_and_indexes = {
            mod: self.aes[mod].latent_batch(x[mod]) for mod in self.modalities
        }
        latents = np.concatenate(
            [latents_and_indexes[mod][0] for mod in self.modalities], axis=0
        )
        idxes = np.concatenate(
            [latents_and_indexes[mod][1] for mod in self.modalities], axis=0
        )
        return latents, idxes

    def imputation_batch(
        self, x, imp_from: str, imp_to: str, to_batch: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform imputation from one modality to another on a mini-batch after the end of
        the training. The imputation is done by first encoding in the latent space cells
        from the modality imp_from, and then decoding these embeddings into the modality
        imp_to.

        :param x: input data mini_batch.
        :param imp_from: the modality of cells for which we want to perform the
            imputation.
        :param imp_to: the modality of the features we want to impute.
        :param to_batch: the batch of the modality imp_to to use for the imputation. If
            None, use the first batch.
        :return: The imputations and their corresponding observation names to ensure
            proper indexing of the results.
        """
        inference_dic = self.aes[imp_from].inference(x)
        inference_dic["z"] = inference_dic["qz_m"]

        inference_dic["batch_index"] = torch.zeros_like(inference_dic["batch_index"])
        if to_batch is not None:
            batch_val = self.aes[imp_to].batch_dic(to_batch)
            inference_dic["batch_index"] += batch_val

        imputation_dic = self.aes[imp_to].generative(inference_dic)
        imp = imputation_dic["x_r"].detach().cpu().numpy()
        return imp, np.array(x["cell_index"])

    def predict_step(
        self, batch: dict[str, dict[str, torch.Tensor]], batch_idx
    ) -> dict[str, np.ndarray]:
        """
        Perform a prediction step on a mini-batch. The prediction step can be either
        imputation or latent embedding.

        :param batch: the input data mini-batch. Not to be confused with cell
            experimental batches.
        :param batch_idx: index of the mini-batch in the dataset. Not to be confused
            with cell experimental batches. Not used in this function but required by
            PyTorch Lightning.
        :return: dictionary with results of predictions and their corresponding
            observation names to ensure proper indexing.
        """
        batch = format_batch(batch)
        if self.predict_mode == "imputation":
            imputations, cell_idxes = self.imputation_batch(
                batch,
                imp_from=self.impute_from,
                imp_to=self.impute_to,
                to_batch=self.to_batch,
            )
            results = {"imputations": imputations, "cell_idxes": cell_idxes}
            return results
        else:
            latents, cell_idxes = self.latent_batch(batch)
            results = {"latents": latents, "cell_idxes": cell_idxes}
            return results

    def get_imputation(
        self,
        impute_from: str,
        impute_to: str,
        to_batch: str | None = None,
        use_cuda: bool = True,
        batch_size: int = 512,
        **dl_kwargs,
    ) -> pd.DataFrame:
        """
        Perform imputation from one modality to another on the whole dataset after the
        end of the training.

        :param impute_from: the modality of cells for which we want to perform the
            imputation.
        :param impute_to: the modality of the features we want to impute.
        :param to_batch: the batch of the modality impute_to to use for the imputation.
            If None, use the first batch.
        :param use_cuda: whether to use GPU acceleration if cuda is available.
        :param batch_size: size of the mini-batches used for the imputation.
        :param dl_kwargs: additional keyword arguments for the DataLoader.
        :return: The imputations a dataFrame indexed by their observation names from the
            input MuData.
        """
        error_message = ""
        if not self.trained:
            error_message += (
                "Model has not been trained yet. Use .fit() to train the model before "
                "imputing. If you want the imputations nonetheless, set the attribute "
                "<trained> to true manually for the  model and rerun this function.\n"
            )
        if impute_from not in self.modalities:
            error_message += (
                f"impute_from should be one of the following "
                f"modalities: {self.modalities}\n"
            )
        if impute_to not in self.modalities:
            error_message += (
                f"impute_to should be one of the following "
                f"modalities: {self.modalities}\n"
            )
        if impute_from == impute_to:
            error_message += (
                "impute_from and impute_to should be different modalities.\n"
            )
        if error_message != "":
            raise ValueError(error_message)

        loader, trainer = inference_dl_trainer(
            self.aes[impute_from].dataset,
            use_cuda=use_cuda,
            batch_size=batch_size,
            **dl_kwargs,
        )
        self.predict_mode = "imputation"
        self.impute_from = impute_from
        self.impute_to = impute_to
        self.to_batch = to_batch
        predict_results = trainer.predict(model=self, dataloaders=loader)

        imputations = pd.DataFrame(
            np.concatenate(
                [batch_res_dic["imputations"] for batch_res_dic in predict_results]
            ),
            index=np.concatenate(
                [batch_res_dic["cell_idxes"] for batch_res_dic in predict_results]
            ),
        )
        return imputations

    def get_iot_loss(
        self, z_1: torch.Tensor, z_2: torch.Tensor, c_cross: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the IOT loss between two sets of latent embeddings.

        :param z_1: latent embeddings from the first modality.
        :param z_2: latent embeddings from the second modality.
        :param c_cross: cost matrix between the two modalities. Rows correspond to z_1
            and columns to z_2.
        :return: The IOT loss.
        """
        p = ot.unif(z_1.size(0), type_as=c_cross)
        q = ot.unif(z_2.size(0), type_as=c_cross)
        c_z = ot.dist(z_1, z_2, metric="euclidean")
        plan = ot.partial.partial_wasserstein(
            p.cpu(), q.cpu(), c_cross.cpu(), m=self.mass
        )
        plan = plan.to(c_cross.device)
        iot_loss = torch.sum(plan * c_z)
        return iot_loss

    def get_sinkhorn_reg(self, z_1: torch.Tensor, z_2: torch.Tensor) -> torch.Tensor:
        """
        Compute the Sinkhorn regularization term between two sets of latent embeddings.

        :param z_1: latent embeddings from the first modality.
        :param z_2: latent embeddings from the second modality.
        :return: the unbalanced Sinkhorn regularization term.
        """
        mixing_module = SamplesLoss(
            loss="sinkhorn", p=2, scaling=0.1, blur=self.blur, reach=self.reach
        )
        mixing_loss = mixing_module(z_1, z_2)
        return mixing_loss

    def loss(
        self,
        x: dict[str, dict[str, torch.Tensor]],
        inference_output: dict[str, dict[str, torch.Tensor]],
        generative_output: dict[str, dict[str, torch.Tensor]],
        reduce: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all the losses for the model.

        :param x: input data mini_batch. A dictionary of dictionaries, where the first
            level correspond to the modalities and the second level correspond to the
            data for each modality.
        :param inference_output: The results of the encoding of input data for each
            modality in a dictionary with the same structure as the input data. Contains
            the latent embeddings.
        :param generative_output: The results of the decoding of the inferred embeddings
            for each modality in a dictionary with the same structure as the input data.
            Contains the reconstruction of the data.
        :param reduce: whether to reduce the cell reconstruction losses to a single
            scalar or not.
        :return: The weighted sum of all sum terms which constitute the final loss.
        """
        unimodal_losses = {
            mod: self.aes[mod].loss(
                x[mod],
                generative_output=generative_output[mod],
                inference_output=inference_output[mod],
                reduce=reduce,
            )
            for mod in self.modalities
        }

        loss_dic = {}
        for mod in self.modalities:
            for key, val in unimodal_losses[mod].items():
                loss_dic[key + "_" + mod] = val
            self.metrics_to_log += self.aes[mod].metrics_to_log

        iot_loss = torch.tensor([0.0], device=x[self.modalities[0]]["input"].device)
        sinkhorn_loss = torch.tensor(
            [0.0], device=x[self.modalities[0]]["input"].device
        )
        for mod_pair in self.modality_pairs:
            c_cross = x["cross"][mod_pair]
            mod1 = mod_pair.split("+")[0]
            mod2 = mod_pair.split("+")[1]
            z_1 = inference_output[mod1]["z"]
            z_2 = inference_output[mod2]["z"]
            iot_loss = self.get_iot_loss(z_1=z_1, z_2=z_2, c_cross=c_cross)
            self.metrics_to_log.append(
                {
                    "name": "iot_loss_{}".format(mod_pair),
                    "value": iot_loss,
                    "batch_size": 1,
                    "reduce_fx": torch.mean,
                }
            )
            sinkhorn_loss = self.get_sinkhorn_reg(z_1=z_1, z_2=z_2)
            self.metrics_to_log.append(
                {
                    "name": "sinkhorn_reg_{}".format(mod_pair),
                    "value": sinkhorn_loss,
                    "batch_size": 1,
                    "reduce_fx": torch.mean,
                }
            )
            iot_loss += iot_loss
            sinkhorn_loss += sinkhorn_loss
        loss_dic["iot_loss"] = self.iot_loss_weight * iot_loss
        loss_dic["sinkhorn_reg"] = self.sinkhorn_loss_weight * sinkhorn_loss
        return loss_dic

    def log_latent_norms(
        self,
        inference_dic: dict[str, dict[str, torch.Tensor]],
        return_log: bool = False,
    ):
        """
        Log the norms of the latent embeddings for each modality.

        :param inference_dic: The results of the encoding of input data for each
            modality in a dictionary with the same structure as the input data. Contains
            the latent embeddings.
        :param return_log: unused parameter
        """
        for mod in self.modalities:
            self.metrics_to_log.append(
                self.aes[mod].log_latent_norms(inference_dic[mod], return_log=True)
            )

    def reset_log_metrics(self):
        """
        Reset the metrics to log at the beginning of each new optimization step on a
        mini-batch.
        """
        self.metrics_to_log = []
        for mod in self.modalities:
            self.aes[mod].reset_log_metrics()
