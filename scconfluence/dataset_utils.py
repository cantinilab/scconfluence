import anndata as ad
import mudata as md
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Union
from scipy.sparse import issparse
import pytorch_lightning as pl


class DictDataset(Dataset):
    """
    Dataset class for dictionary data. It is a subclass of torch.utils.data.Dataset. It
    is meant to be used to load data from a dictionary where all entries are indexed by
    the same observations. The keys of the dictionary are the fields and the values are
    the data to be loaded.

    :param dict_data: dictionary with the data.
    """

    def __init__(self, dict_data):
        self.dict_data = dict_data

    def __getitem__(self, index: np.ndarray) -> dict:
        """
        Get the data for a given index.

        :param index: index of the observations to be retrieved.
        :return: a dictionary with the data for the given index.
        """
        out = {}
        for key, value in self.dict_data.items():
            if key == "cell_index":
                out[key] = list(np.array(value)[index])
            else:
                if issparse(value):
                    out[key] = torch.tensor(value[index].todense()).float()
                else:
                    out[key] = value[index]
        return out

    def split_train_val(self, ratio_val: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset into a training and a validation set.

        :param ratio_val: proportion of cells to be used for validation.
        :return: the indices of the observations to be used for training and validation.
        """
        n = len(self)
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = int(n * ratio_val)
        idx_val = idx[:n_val]
        idx_train = idx[n_val:]
        return idx_train, idx_val

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: the number of observations in the dataset.
        """
        key = "batch_index"
        return len(self.dict_data[key])


class MModalDataset(Dataset):
    """
    Dataset class for multi-modal data. It is a subclass of torch.utils.data.Dataset.

    :param datasets: dictionary with the data for each modality. The keys are the names
        of the modalities and the values are the data for each modality stored in
        DictDataset objects.
    :param cross_relations: dictionary with the cross-modal relations. The keys are the
        names of the pair of modalities compared and the values are the cost matrices.
    """

    def __init__(
        self, datasets: dict[str, DictDataset], cross_relations: dict[str, np.ndarray]
    ):
        self.datasets = datasets
        self.modalities = list(self.datasets.keys())
        self.cross_relations = cross_relations
        self.idx_intervals = {
            self.modalities[0]: [0, len(self.datasets[self.modalities[0]])]
        }
        for i in range(1, len(self.modalities)):
            self.idx_intervals[self.modalities[i]] = [
                self.idx_intervals[self.modalities[i - 1]][1],
                self.idx_intervals[self.modalities[i - 1]][1]
                + len(self.datasets[self.modalities[i]]),
            ]

    def __getitem__(self, index: np.ndarray) -> dict:
        """
        Get the data for a given index.

        :param index: index of the observations to be retrieved.
        :return: a dictionary with the data for the given index.
        """
        index = np.array(index)
        indexes = {}
        for mod in self.modalities:
            indexes[mod] = (
                index[
                    np.logical_and(
                        index >= self.idx_intervals[mod][0],
                        index < self.idx_intervals[mod][1],
                    )
                ]
                - self.idx_intervals[mod][0]
            )
        out = {
            mod: self.datasets[mod].__getitem__(indexes[mod]) for mod in self.modalities
        }
        if self.cross_relations is not None:
            out["cross"] = {
                modality_pair: torch.tensor(
                    cost[indexes[modality_pair.split("+")[0]]][
                        :, indexes[modality_pair.split("+")[1]]
                    ]
                )
                for modality_pair, cost in self.cross_relations.items()
            }
        return out

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: the number of observations in the dataset.
        """
        return sum([len(self.datasets[mod]) for mod in self.modalities])

    def split_train_val(self, ratio_val: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset into a training and a validation set.

        :param ratio_val: proportion of cells to be used for validation.
        :return: the indices of the observations to be used for training and validation.
        """
        idx_train, idx_val = [], []
        for mod, ds in self.datasets.items():
            idx_train_mod, idx_val_mod = ds.split_train_val(ratio_val)
            idx_train.append(idx_train_mod + self.idx_intervals[mod][0])
            idx_val.append(idx_val_mod + self.idx_intervals[mod][0])
        return np.concatenate(idx_train), np.concatenate(idx_val)


class BatchSampler(torch.utils.data.sampler.Sampler):
    """
    Code imported and adapted from
    https://github.com/scverse/scvi-tools/blob/master/scvi/dataloaders/_ann_dataloader.py
    Custom torch Sampler that returns a list of indices of size batch_size.
    """

    def __init__(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: Union[bool, int] = False,
    ):
        self.indices = indices
        self.n_obs = len(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last > batch_size:
            raise ValueError(
                "drop_last can't be greater than batch_size. "
                + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
            )

        last_batch_len = self.n_obs % self.batch_size
        if (drop_last is True) or (last_batch_len < drop_last):
            drop_last_n = last_batch_len
        elif (drop_last is False) or (last_batch_len >= drop_last):
            drop_last_n = 0
        else:
            raise ValueError("Invalid input for drop_last param. Must be bool or int.")

        self.drop_last_n = drop_last_n

    def __iter__(self):
        if self.shuffle is True:
            idx = torch.randperm(self.n_obs).tolist()
        else:
            idx = torch.arange(self.n_obs).tolist()

        if self.drop_last_n != 0:
            idx = idx[: -self.drop_last_n]

        data_iter = iter(
            [
                self.indices[idx[i : i + self.batch_size]]
                for i in range(0, len(idx), self.batch_size)
            ]
        )
        return data_iter

    def __len__(self):
        from math import ceil

        if self.drop_last_n != 0:
            length = self.n_obs // self.batch_size
        else:
            length = ceil(self.n_obs / self.batch_size)
        return length


def format_batch(batch: dict) -> dict:
    """
    Format the batch to be used in the model. It removes the batch dimension from the
    data and the cell index.

    :param batch: mini-batch of input data.
    :return: formatted mini-batch.
    """
    for key, value in batch.items():
        try:
            if key == "cell_index":
                batch[key] = [n[0] for n in value]
            else:
                batch[key] = value.squeeze(0)
        except AttributeError:
            for entry, val in value.items():
                if entry == "cell_index":
                    batch[key][entry] = [n[0] for n in val]
                else:
                    batch[key][entry] = val.squeeze(0)
    return batch


def configure_unimodal_dataset(
    adata: ad.AnnData,
    rep_in: None | str,
    rep_out: None | str,
    batch_key: str,
    modality: str,
) -> tuple[DictDataset, int, int, dict[str, int]]:
    """
    Create a dictionary dataset from an anndata.

    :param adata: data
    :param rep_in: string indicating the entry of the Anndata where to look for the
        input data, i.e. the data used as input of the encoder. If not None, the input
        data will be extracted from the obsm field of the AnnData object. If None, the
        input data is assumed to be  the X field of the AnnData object.
    :param rep_out: string indicating the entry of the Anndata where to look for the
        output data, i.e. the data used to compare with the output of the decoder. If
        not None, the output data will be extracted from the layers field of the AnnData
        object. If None, the output data is assumed to be  the X field of the AnnData
        object.
    :param batch_key: where to extract the batch information in the adata object
    :param modality: name of the data modality
    :return: a DictDataset object, the input dimension, the output dimension and a
        dictionary mapping batch indexes to their original name.
    """

    def get_anndata_entry(field: str, key: str):
        if field == "layers":
            if key not in adata.layers.keys():
                raise ValueError(f"Layer <{key}> not found in adata of <{modality}>")
            return adata.layers[key]
        elif field == "obsm":
            if key not in adata.obsm.keys():
                raise ValueError(f"Obsm <{key}> not found in adata of <{modality}>")
            return adata.obsm[key]
        elif field == "X":
            return adata.X
        elif field == "obs":
            if key not in adata.obs.keys():
                raise ValueError(f"Obs <{key}> not found in adata of <{modality}>")
            return adata.obs[key].values
        else:
            raise ValueError(
                f"Field <{field}> not recognized in registry of <{modality}>"
            )

    dic_data = {"cell_index": list(adata.obs_names)}
    for prefix, key, field in zip(
        ["input", "output"], [rep_in, rep_out], ["obsm", "layers"]
    ):
        if key is not None:
            extracted_data = get_anndata_entry(field=field, key=key)
        else:
            extracted_data = get_anndata_entry(field="X", key="X")
        if issparse(extracted_data):
            dic_data[prefix] = extracted_data
        else:
            dic_data[prefix] = torch.from_numpy(extracted_data)
    if batch_key is not None:
        batches = get_anndata_entry(field="obs", key=batch_key)
    else:
        batches = ["batch" for _ in range(adata.n_obs)]
    map_batch_index = {val: k for k, val in enumerate(set(batches))}
    batch_indexes = torch.tensor([map_batch_index[b] for b in batches]).unsqueeze(1)
    dic_data["batch_index"] = batch_indexes
    return (
        DictDataset(dic_data),
        dic_data["input"].shape[1],
        dic_data["output"].shape[1],
        map_batch_index,
    )


def configure_multimodal_dataset(
    mdata: md.MuData,
    modality_pairs: list[str],
    unimodal_datasets: dict[str, DictDataset],
) -> MModalDataset:
    """
    Create the MModalDataset object from the MuData object

    :param mdata: the input data
    :param modality_pairs: the pairs of modalities for which a cost matrix is available
    :param unimodal_datasets: the unimodal datasets
    :return: a MModalDataset which can be used for training and inference
    """
    cross_relations = {
        mod_pair: mdata.uns[f"cross_{mod_pair}"] for mod_pair in modality_pairs
    }
    return MModalDataset(
        datasets={mod: dataset for mod, dataset in unimodal_datasets.items()},
        cross_relations=cross_relations,
    )


def inference_dl_trainer(
    dataset: DictDataset | MModalDataset,
    use_cuda: bool = True,
    batch_size: int = 512,
    pin_memory: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, pl.Trainer]:
    """
    Wrapper to create a DataLoader and a Trainer for the prediction after the end of the
    training of the model.

    :param dataset: dataset to be used for the prediction (which has also benn used for
        the training).
    :param use_cuda: whether to use GPU acceleration if cuda is available.
    :param batch_size: size of the mini-batches used for training. Not to be confused
        with the experimental batches.
    :param pin_memory: If True, the data loader will copy Tensors into device/CUDA
        pinned memory before returning them.
    :param num_workers: how many subprocesses to use for data loading. 0 means that the
        data will be loaded in the main process.
    :return: a DataLoader and a Trainer for the prediction.
    """
    loader = DataLoader(
        dataset,
        sampler=BatchSampler(
            indices=np.arange(len(dataset)),
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
        ),
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if (use_cuda and torch.cuda.is_available()) else "cpu",
        devices=1,
    )
    return loader, trainer
