from scconfluence import model, unimodal
import anndata as ad
import numpy as np
import mudata as md


def create_synthetic_adata(n_obs, n_features, n_batch, dim_pca, obs_prefix=""):
    """
    Create a synthetic AnnData object with the specified number of observations, features and batches.
    :param n_obs: number of observations
    :param n_features: number of features
    :param n_batch: number of batches
    :param dim_pca: number of dimensions for PCA
    :param obs_prefix: prefix for observation names (useful when generating multiple modalities)
    :return: synthetic AnnData object
    """
    adata = ad.AnnData(
        X=np.random.rand(n_obs, n_features).astype(np.float32),
        obs={"batches": np.random.choice(range(n_batch), n_obs)},
        layers={
            "counts": np.random.randint(
                low=0, high=100, size=(n_obs, n_features)
            ).astype(np.float32),
            "binary": np.random.randint(low=0, high=2, size=(n_obs, n_features)).astype(
                np.float32
            ),
        },
        obsm={"X_pca": np.random.rand(n_obs, dim_pca).astype(np.float32)},
    )
    adata.obs_names = [f"{obs_prefix}obs_{i}" for i in range(n_obs)]
    return adata


def create_synthetic_mdata(dic_obs, dic_features, dic_batch, dim_pca=20):
    """
    Create a synthetic MuData object with the specified number of observations, features, batches and
    modalities.
    :param dic_obs: number of observations
    :param dic_features: number of features
    :param dic_batch: number of batches
    :param dim_pca: number of dimensions for PCA
    :return: synthetic MuData object
    """
    if set(dic_obs.keys()) != set(dic_features.keys()) or set(dic_obs.keys()) != set(
        dic_batch.keys()
    ):
        raise ValueError("The keys of the input dictionaries must be the same.")
    modalities = list(dic_obs.keys())
    mdata = md.MuData(
        {
            mod: create_synthetic_adata(
                n_obs=dic_obs[mod],
                n_features=dic_features[mod],
                n_batch=dic_batch[mod],
                dim_pca=dim_pca,
                obs_prefix=f"{mod}_",
            )
            for mod in modalities
        }
    )
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            mdata.uns[f"cross_{modalities[i]}+{modalities[j]}"] = np.random.rand(
                dic_obs[modalities[i]], dic_obs[modalities[j]]
            ).astype(np.float32)
    mdata.uns["cross_keys"] = [
        f"{modalities[i]}+{modalities[j]}"
        for i in range(len(modalities))
        for j in range(i + 1, len(modalities))
    ]
    return mdata


def run_unimodal_ae(
    path,
    n_obs,
    n_features,
    n_batch,
    dim_pca,
    rep_in,
    rep_out,
    type_loss,
    n_hidden,
    n_latent,
    batch_key,
):
    adata = create_synthetic_adata(
        n_obs=n_obs, n_features=n_features, n_batch=n_batch, dim_pca=dim_pca
    )
    ae = unimodal.AutoEncoder(
        adata=adata,
        rep_in=rep_in,
        rep_out=rep_out,
        modality="mod1",
        n_hidden=n_hidden,
        n_latent=n_latent,
        batch_key=batch_key,
        type_loss=type_loss,
    )
    ae.fit(save_path=path, use_cuda=False, ratio_val=0.2, batch_size=32, max_epochs=2)
    adata.obsm["latent_embeddings"] = (
        ae.get_latent(use_cuda=False, batch_size=25).loc[adata.obs_names].values
    )
    assert adata.obsm["latent_embeddings"].shape == (n_obs, n_latent)


def run_scconfluence(
    path,
    mdata,
    unimodal_aes,
    impute_from,
    impute_to,
    dic_obs,
    dic_features,
    n_latent,
):
    our_model = model.ScConfluence(
        mdata=mdata,
        unimodal_aes=unimodal_aes,
        mass=0.5,
        reach=1.0,
        iot_loss_weight=0.05,
        sinkhorn_loss_weight=0.1,
    )
    our_model.fit(
        save_path=path, use_cuda=False, ratio_val=0.2, batch_size=100, max_epochs=2
    )
    mdata.obsm["latent_embeddings"] = (
        our_model.get_latent(use_cuda=False, batch_size=100).loc[mdata.obs_names].values
    )
    assert mdata.obsm["latent_embeddings"].shape == (sum(dic_obs.values()), n_latent)

    mdata[impute_from].obsm["imputations"] = (
        our_model.get_imputation(
            impute_from=impute_from, impute_to=impute_to, use_cuda=False, batch_size=100
        )
        .loc[mdata[impute_from].obs_names]
        .values
    )
    assert mdata[impute_from].obsm["imputations"].shape == (
        dic_obs[impute_from],
        dic_features[impute_to],
    )
