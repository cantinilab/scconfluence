import collections
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

# A few helper functions here have been copied from the scvi-tools repo and adapted for our use in order to avoid
# adding scvi-tools to the installation requirements of this package as scvi-tools has many requirements.
# Additionally, here we used the DSBN implementation from https://github.com/woozch/DSBN


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
                for i in range(n_domain)
            ]
        )

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(
            x.size(0), self.num_features, device=x.device
        )  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                # out[indices] = x[indices]
                self.bns[i].training = False
                out[indices] = self.bns[i](x[indices])
                self.bns[i].training = True
        return out


class VariationalEncoder(nn.Module):
    """
    Taken from https://github.com/YosefLab/scvi-tools/
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    cst_noise
        Whether to only use the hardcoded constant var_eps for the variance of the variational distribution
    **kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_batch: int = 1,
        deeply_inject_covariates: bool = True,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-4,
        var_activation=None,
        **kwargs,
    ):
        super().__init__()

        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers - 1,
            n_hidden=n_hidden,
            n_batch=n_batch,
            deeply_inject_covariates=deeply_inject_covariates,
            dropout_rate=dropout_rate,
            positive_out=False,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor):
        r"""
        The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        batch_index
            tensor
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        # Parameters for latent distribution
        q = self.encoder(x, batch_index)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


class DecoderPeakVI(torch.nn.Module):
    """
    Taken from https://github.com/YosefLab/scvi-tools/
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 2,
        n_hidden: int = 128,
        n_batch: int = 1,
        use_batch_norm: str = None,
        deeply_inject_covariates: bool = True,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_batch=n_batch,
            deeply_inject_covariates=deeply_inject_covariates,
            activation_fn=nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            last_activation=True,
            dropout_rate=0.0,
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, batch_index: torch.Tensor):
        x = self.output(self.px_decoder(z, batch_index))
        return x


class DecoderSCVI(nn.Module):
    """
    Taken from https://github.com/YosefLab/scvi-tools/
    Decodes data from latent space of ``n_input`` dimensions into ``n_output``dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_batch: int = 1,
        use_batch_norm: str = None,
        deeply_inject_covariates: bool = True,
        use_poisson: bool = False,
    ):
        super().__init__()
        self.use_poisson = use_poisson
        self.px_decoder = FCLayers(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_batch=n_batch,
            use_batch_norm=use_batch_norm,
            deeply_inject_covariates=deeply_inject_covariates,
            last_activation=True,
            dropout_rate=0.0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        if not use_poisson:
            # dispersion: here we only deal with gene-cell dispersion case
            self.px_r_decoder = nn.Linear(n_hidden, n_output)

            # dropout
            self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        log_library: torch.Tensor,
        batch_index: torch.Tensor,
    ):
        """
        The forward computation for a single sample.
         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``
        Parameters
        ----------
        dispersion
            One of the following
            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        log_library
            log library size
        batch_index
            tensor of batch membership(s) for this sample
        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, batch_index)
        px_scale = self.px_scale_decoder(px)
        if self.use_poisson:
            px_dropout = None
            px_r = None
        else:
            px_dropout = self.px_dropout_decoder(px)
            px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(log_library) * px_scale  # torch.clamp( , max=12)
        return px_scale, px_r, px_rate, px_dropout


def get_batch_norm(flag, n_out, n_batch):
    if flag is None:
        return None
    elif flag == "standard" or (flag == "ds" and n_batch == 0):
        return nn.BatchNorm1d(
            n_out, eps=1e-5, momentum=0.1
        )  # momentum=0.01, eps=0.001)
    elif flag == "ds" and n_batch > 0:
        return DSBatchNorm(n_out, n_batch, eps=1e-5, momentum=0.1)
    else:
        raise ValueError("Unrecognized <use_batch_norm> value: {}".format(flag))


class FCLayers(nn.Module):
    """
    Taken from https://github.com/YosefLab/scvi-tools/
    """

    def __init__(
        self,
        n_layers,
        n_input,
        n_hidden,
        n_output,
        n_batch=1,
        deeply_inject_covariates=True,
        use_batch_norm=None,
        activation_fn=nn.ReLU,
        last_activation=False,
        positive_out=False,
        dropout_rate=0.0,
        bias=True,
    ):
        super().__init__()
        self.deeply_inject_covariates = deeply_inject_covariates
        if n_batch > 1:
            self.n_batch = n_batch
        else:
            self.n_batch = 0
        self.positive_output = positive_out
        layers_dim = [n_input] + (n_layers - 1) * [n_hidden] + [n_output]

        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i + 1),
                        nn.Sequential(
                            nn.Linear(
                                n_in + self.n_batch * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            get_batch_norm(use_batch_norm, n_out, n_batch),
                            (
                                activation_fn()
                                if i < n_layers - 1 or last_activation
                                else None
                            ),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.deeply_inject_covariates)
        return user_cond

    def forward(self, x, batch_index):
        if self.n_batch > 0:
            one_hot_batch = one_hot(batch_index, self.n_batch)
        for i, layer in enumerate(self.layers):
            for layer_component in layer:
                if layer_component is not None:
                    if isinstance(layer_component, DSBatchNorm):
                        x = layer_component(x, batch_index)
                    else:
                        if (
                            isinstance(layer_component, nn.Linear)
                            and self.inject_into_layer(i)
                            and self.n_batch > 0
                        ):
                            x = torch.cat((x, one_hot_batch), dim=-1)
                        x = layer_component(x)
        if self.positive_output:
            x = nn.ReLU()(x)
        return x
