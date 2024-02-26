from scconfluence import dataset_utils, unimodal
import numpy as np
from torch.utils.data import DataLoader
from context import (
    create_synthetic_adata,
    create_synthetic_mdata,
    run_unimodal_ae,
    run_scconfluence,
)


def test_umodal_dataset():
    adata = create_synthetic_adata(n_obs=100, n_features=200, n_batch=3, dim_pca=20)
    dataset, d_input, d_output, batch_dic = dataset_utils.configure_unimodal_dataset(
        adata=adata, batch_key="batches", rep_in="X_pca", rep_out=None, modality="mod1"
    )
    assert len(batch_dic) == 3
    assert d_input == 20
    assert d_output == 200
    assert len(dataset) == 100
    assert dataset.__getitem__(np.array([0, 1]))["input"].shape == (2, 20)
    assert dataset.__getitem__(np.array([0, 1]))["output"].shape == (2, 200)

    # create data loader from dataset
    dl = DataLoader(
        dataset,
        sampler=dataset_utils.BatchSampler(
            indices=np.arange(len(dataset)),
            batch_size=40,
            shuffle=True,
            drop_last=False,
        ),
    )
    for _ in dl:
        pass


def test_mmodal_dataset():
    mdata = create_synthetic_mdata(
        dic_obs={"mod1": 100, "mod2": 150},
        dic_features={"mod1": 30, "mod2": 300},
        dic_batch={"mod1": 3, "mod2": 4},
        dim_pca=20,
    )
    unimodal_datasets = {
        mod: dataset_utils.configure_unimodal_dataset(
            adata=mdata[mod],
            batch_key="batches",
            rep_in=None,
            rep_out=None,
            modality=mod,
        )[0]
        for mod in mdata.mod.keys()
    }
    modality_pairs = [name.split("cross_")[-1] for name in mdata.uns["cross_keys"]]
    dataset = dataset_utils.configure_multimodal_dataset(
        mdata=mdata, unimodal_datasets=unimodal_datasets, modality_pairs=modality_pairs
    )
    assert len(dataset) == 250

    # create data loader from dataset
    dl = DataLoader(
        dataset,
        sampler=dataset_utils.BatchSampler(
            indices=np.arange(len(dataset)),
            batch_size=40,
            shuffle=True,
            drop_last=False,
        ),
    )
    for _ in dl:
        pass


def test_l2_ae(tmp_path):
    run_unimodal_ae(
        path=tmp_path,
        n_obs=100,
        n_features=200,
        n_batch=3,
        dim_pca=20,
        rep_in=None,
        rep_out=None,
        type_loss="l2",
        n_hidden=15,
        n_latent=10,
        batch_key="batches",
    )


def test_zinb_ae(tmp_path):
    run_unimodal_ae(
        path=tmp_path,
        n_obs=100,
        n_features=200,
        n_batch=3,
        dim_pca=20,
        rep_in="X_pca",
        rep_out="counts",
        type_loss="zinb",
        n_hidden=15,
        n_latent=10,
        batch_key="batches",
    )


def test_nb_ae(tmp_path):
    run_unimodal_ae(
        path=tmp_path,
        n_obs=100,
        n_features=200,
        n_batch=3,
        dim_pca=20,
        rep_in="X_pca",
        rep_out="counts",
        type_loss="nb",
        n_hidden=15,
        n_latent=10,
        batch_key="batches",
    )


def test_poisson_ae(tmp_path):
    run_unimodal_ae(
        path=tmp_path,
        n_obs=100,
        n_features=200,
        n_batch=3,
        dim_pca=20,
        rep_in="X_pca",
        rep_out="counts",
        type_loss="poisson",
        n_hidden=15,
        n_latent=10,
        batch_key="batches",
    )


def test_binary_ae(tmp_path):
    run_unimodal_ae(
        path=tmp_path,
        n_obs=100,
        n_features=200,
        n_batch=3,
        dim_pca=20,
        rep_in="X_pca",
        rep_out="binary",
        type_loss="binary",
        n_hidden=15,
        n_latent=10,
        batch_key="batches",
    )


def test_batchnorm_ae(tmp_path):
    run_unimodal_ae(
        path=tmp_path,
        n_obs=100,
        n_features=200,
        n_batch=3,
        dim_pca=20,
        rep_in="X_pca",
        rep_out="binary",
        type_loss="binary",
        n_hidden=15,
        n_latent=10,
        batch_key="batches",
        use_batch_norm_enc="standard",
        use_batch_norm_dec="ds",
    )


def test_2modalities(tmp_path):
    dic_obs = {"mod1": 100, "mod2": 150}
    dic_features = {"mod1": 30, "mod2": 300}
    n_latent = 10
    mdata = create_synthetic_mdata(
        dic_obs=dic_obs,
        dic_features=dic_features,
        dic_batch={"mod1": 3, "mod2": 4},
        dim_pca=20,
    )
    unimodal_aes = dict()
    unimodal_aes["mod1"] = unimodal.AutoEncoder(
        adata=mdata["mod1"],
        rep_in=None,
        rep_out=None,
        modality="mod1",
        n_hidden=15,
        n_latent=n_latent,
        batch_key="batches",
        type_loss="l2",
    )
    unimodal_aes["mod2"] = unimodal.AutoEncoder(
        adata=mdata["mod2"],
        rep_in="X_pca",
        rep_out="counts",
        modality="mod2",
        n_hidden=50,
        n_latent=n_latent,
        batch_key="batches",
        type_loss="zinb",
    )
    run_scconfluence(
        path=tmp_path,
        mdata=mdata,
        unimodal_aes=unimodal_aes,
        impute_from="mod1",
        impute_to="mod2",
        dic_obs=dic_obs,
        dic_features=dic_features,
        n_latent=n_latent,
    )


def test_3omics_mmae(tmp_path):
    dic_obs = {"mod1": 100, "mod2": 150, "mod3": 120}
    dic_features = {"mod1": 30, "mod2": 300, "mod3": 50}
    dic_batch = {"mod1": 3, "mod2": 4, "mod3": 2}
    dic_hidden = {"mod1": 15, "mod2": 50, "mod3": 150}
    n_latent = 10
    mdata = create_synthetic_mdata(
        dic_obs=dic_obs,
        dic_features=dic_features,
        dic_batch=dic_batch,
        dim_pca=20,
    )
    unimodal_aes = dict()
    unimodal_aes["mod1"] = unimodal.AutoEncoder(
        adata=mdata["mod1"],
        rep_in=None,
        rep_out=None,
        modality="mod1",
        n_hidden=dic_hidden["mod1"],
        n_latent=n_latent,
        batch_key="batches",
        type_loss="l2",
    )
    unimodal_aes["mod2"] = unimodal.AutoEncoder(
        adata=mdata["mod2"],
        rep_in="X_pca",
        rep_out="counts",
        modality="mod2",
        n_hidden=dic_hidden["mod2"],
        n_latent=n_latent,
        batch_key="batches",
        type_loss="zinb",
    )
    unimodal_aes["mod3"] = unimodal.AutoEncoder(
        adata=mdata["mod3"],
        rep_in="X_pca",
        rep_out=None,
        modality="mod3",
        n_hidden=dic_hidden["mod3"],
        n_latent=n_latent,
        batch_key="batches",
        type_loss="nb",
    )
    run_scconfluence(
        path=tmp_path,
        mdata=mdata,
        unimodal_aes=unimodal_aes,
        impute_from="mod2",
        impute_to="mod3",
        dic_obs=dic_obs,
        dic_features=dic_features,
        n_latent=n_latent,
    )
