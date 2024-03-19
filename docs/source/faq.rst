Frequently Asked Questions
--------------------------

If you encounter an issue or have a question that is not addressed in this FAQ,
please open an issue on the `github repository <https://github.com/cantinilab/scconfluence>`_.

What should I do if the inferred latent embeddings from different modalities are completely separated?
""""""""""""""""""""
There are two parameters that can be adjusted to avoid this issue. The first one is the
`sinkhorn_loss_weight` parameter, which controls the weight of the Sinkhorn loss in the
overall loss function. Increasing it should lead to a more mixed latent space but this
might come at the expense of the reconstruction quality. The second parameter is the
`reach` parameter which controls the unbalancedness of the sinkhorn regularization.
Increasing it will lead to a more less mass being discarded from the sinkhorn
regularization's transport plan and therefore to a more mixed latent space.

What should I do if the inferred cell embeddings from one (or more) modality don't seem to preserve their structure?
""""""""""""""""""""

If you have access to cell labels for one modality, you
might want to plot those information on top of the UMAP of the inferred embeddings to
verify that the biological structure of the dataset is preserved. If it is not, there
could be two possible reasons why the structure was lost:

1. The unimodal autoencoder or the preprocessing that you use are not adapted to the \
data. To investigate whether this is the case you could train a single autoencoder on \
the measurements from the modality in question and verify that the structure is \
preserved in the unimodal embeddings. To do so, you can use the `.fit()` method to train
one of the autoencoders on its own (as you would for the multimodal `ScConfluence` model).

2. The mixing constraint enforced in the embedding space is too strong and distorts \
too much the biological structure of the embeddings. Decreasing the \
`sinkhorn_loss_weight` parameter might help in this case.