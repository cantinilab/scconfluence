# scConfluence
scConfluence is a novel method for the integration of unpaired multiomics data combining
uncoupled autoencoders and Inverse Optimal Transport to learn low-dimensional cell 
embeddings. These embeddings can then be used for visualization and clustering, useful 
for discovering  subpopulations of cells, and for imputation of features across 
modalities.
Read the preprint! (!!!!add link!!!!)

![figure](model.png)

## Install the package

scConfluence is implemented as a Python package seamlessly integrated within the scverse 
ecosystem, in particular Muon and Scanpy.

### via PyPI (!!!!!!!!! not yet available!!!!!!!)

On all operating systems, the easiest way to install scConfluence is via PyPI. 
Installation should typically take a minute and is continuously tested with Python 3.10 
on an Ubuntu virtual machine.

```bash
pip install scconfluence
```

### via GitHub (development version)

```bash
git clone git@github.com:cantinilab/scconfluence.git
cd scconfluence
pip install .
```

### Test your development installation (optional)

```bash
pip install pytest
pytest .
```

## Getting started

scConfluence takes as an input a MuData object and populates its `obsm` field 
with the latent embeddings. Visit (read the docs page currently under preparation) for 
more documentation and tutorials (which can be found now in the tutorials folder of 
this repository).

You may download a preprocessed 10X Multiome demo dataset [here](https://figshare.com/s/b0840d90e42e37fa165f).

A GPU is not required for the method to run, but is strongly recommended.

```python
import scconfluence
import mudata as md
import scanpy as sc
from scipy.spatial.distance import cdist

# Load data into a Muon object.
mdata = md.read_h5mu("my_data.h5mu")

# Compute the cross-modality distance matrix using connected features
mdata.uns["cross_modality1+modality2"] = cdist(mdata["modality1"].obsm["cm_features"], 
                                               mdata["modality2"].obsm["cm_features"])
mdata.uns["cross_keys"] = ["cross_modality1+modality2"]


# Initialize and train the model.
autoencoders = {"modality1": scconfluence.model.AutoEncoder(mdata["modality1"],
                                                             modality="modality1"),
                "modality2": scconfluence.model.AutoEncoder(mdata["modality2"],
                                                             modality="modality2")}
model = scconfluence.model.ScConfluence(mdata, unimodal_aes=autoencoders)
model.fit(save_path="results")
mdata.obsm["latent"] = model.get_latent().loc[mdata.obs_names]

# Visualize the embedding with UMAP.
sc.pp.neighbors(mdata, use_rep="latent")
sc.tl.umap(mdata)
sc.pl.umap(mdata)
```

## Our preprint

```bibtex
!!!! add citation !!!!
```

If you're looking for the repository with code to reproduce the experiments in our 
preprint, [here it is!](https://github.com/cantinilab/scc_reproducibility)