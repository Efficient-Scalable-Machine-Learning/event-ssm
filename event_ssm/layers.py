from flax import linen as nn
import jax
from functools import partial


class EventPooling(nn.Module):
    """
    Subsampling layer for event sequences.
    """
    stride: int = 1
    mode: str = "last"
    eps: float = 1e-6

    def __call__(self, x, integration_timesteps):
        """
        Compute the pooled (L/stride)xH output given an LxH input.
        :param x: input sequence (L, d_model)
        :param integration_timesteps: the integration timesteps for the SSM
        :return: output sequence (L/stride, d_model)
        """
        if self.stride == 1:
            raise ValueError("Stride 1 not supported for pooling")

        else:
            remaining_timesteps = (len(integration_timesteps) // self.stride) * self.stride
            new_integration_timesteps = integration_timesteps[:remaining_timesteps].reshape(-1, self.stride).sum(axis=1)
            x = x[:remaining_timesteps]
            d_model = x.shape[-1]

            if self.mode == 'last':
                x = x[::self.stride]
                return x, new_integration_timesteps
            elif self.mode == 'avgpool':
                x = x.reshape(-1, self.stride, d_model).mean(axis=1)
                return x, new_integration_timesteps
            elif self.mode == 'timepool':
                weight = integration_timesteps[:remaining_timesteps, None] + self.eps
                x = (x * weight).reshape(-1, self.stride, d_model).sum(axis=1)
                x = x / weight.reshape(-1, self.stride, 1).sum(axis=1)
                return x, new_integration_timesteps
            else:
                raise NotImplementedError("Pooling mode: {} not implemented".format(self.stride))


class SequenceStage(nn.Module):
    """
    Defines a block of EventSSM layers with the same hidden size and event-resolution

    :param ssm: the SSM to be used (i.e. S5 ssm)
    :param d_model_in: this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
    :param d_model_out: this is the feature size of the layer outputs
    :param d_ssm: the size of the state space model
    :param ssm_block_size: the block size of the state space model
    :param layers_per_stage: the number of S5 layers to stack
    :param dropout: dropout rate
    :param prenorm: whether to use layernorm before the module or after it
    :param batchnorm: If True, use batchnorm instead of layernorm
    :param bn_momentum: momentum for batchnorm
    :param step_rescale: rescale the integration timesteps by this factor
    :param pooling_stride: stride for pooling
    :param pooling_mode: pooling mode (last, avgpool, timepool)
    :param state_expansion_factor: factor to expand the state space model
    """
    ssm: nn.Module
    discretization: str
    d_model_in: int
    d_model_out: int
    d_ssm: int
    ssm_block_size: int
    layers_per_stage: int
    dropout: float = 0.0
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1

    @nn.compact
    def __call__(self, x, integration_timesteps, train: bool):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input input sequence.

        :param x: input sequence (L, d_input)
        :param integration_timesteps: the integration timesteps for the SSM
        :param train: If True, applies dropout and batch norm from batch statistics
        :return: output sequence (L, d_model), integration_timesteps
        """
        EventSSMLayer = partial(
            SequenceLayer,
            ssm=self.ssm,
            discretization=self.discretization,
            dropout=self.dropout,
            d_ssm=self.d_ssm,
            block_size=self.ssm_block_size,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )

        # first layer with pooling
        x, integration_timesteps = EventSSMLayer(
            d_model_in=self.d_model_in,
            d_model_out=self.d_model_out,
            pooling_stride=self.pooling_stride,
            pooling_mode=self.pooling_mode
        )(x, integration_timesteps, train=train)

        # further layers without pooling
        for l in range(self.layers_per_stage - 1):
            x, integration_timesteps = EventSSMLayer(
                d_model_in=self.d_model_out,
                d_model_out=self.d_model_out,
                pooling_stride=1
            )(x, integration_timesteps, train=train)

        return x, integration_timesteps


class SequenceLayer(nn.Module):
    """
    Defines a single event-ssm layer, with S5 SSM, nonlinearity,
    dropout, batch/layer norm, etc.

    :param ssm: the SSM to be used (i.e. S5 ssm)
    :param discretization: the discretization method to use (zoh, dirac, async)
    :param dropout: dropout rate
    :param d_model_in: the input feature size
    :param d_model_out: the output feature size
    :param d_ssm: the size of the state space model
    :param block_size: the block size of the state space model
    :param prenorm: whether to use layernorm before the module or after it
    :param batchnorm: If True, use batchnorm instead of layernorm
    :param bn_momentum: momentum for batchnorm
    :param step_rescale: rescale the integration timesteps by this factor
    :param pooling_stride: stride for pooling
    :param pooling_mode: pooling mode (last, avgpool, timepool)
    """
    ssm: nn.Module
    discretization: str
    dropout: float
    d_model_in: int
    d_model_out: int
    d_ssm: int
    block_size: int
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"

    @nn.compact
    def __call__(self, x, integration_timesteps, train: bool):
        """
        Compute a layer step

        :param x: input sequence (L, d_model_in)
        :param integration_timesteps: the integration timesteps for the SSM
        :param train: If True, applies dropout and batch norm from batch statistics
        :return: output sequence (L, d_model_out), integration_timesteps
        """
        skip = x

        if self.prenorm:
            norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch') if self.batchnorm else nn.LayerNorm()
            x = norm(x, use_running_average=not train) if self.batchnorm else norm(x)

        # apply state space model
        x = self.ssm(
            H_in=self.d_model_in, H_out=self.d_model_out, P=self.d_ssm, block_size=self.block_size,
            step_rescale=self.step_rescale, discretization=self.discretization,
            stride=self.pooling_stride, pooling_mode=self.pooling_mode
        )(x, integration_timesteps)

        # non-linear activation function
        x1 = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(nn.gelu(x))
        x1 = nn.Dense(self.d_model_out)(x1)
        x = x * nn.sigmoid(x1)
        x = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not train)(x)

        if self.pooling_stride > 1:
            pool = EventPooling(stride=self.pooling_stride, mode=self.pooling_mode)
            skip, integration_timesteps = pool(skip, integration_timesteps)

        if self.d_model_in != self.d_model_out:
            skip = nn.Dense(self.d_model_out)(skip)

        x = skip + x

        if not self.prenorm:
            norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch') if self.batchnorm else nn.LayerNorm()
            x = norm(x, use_running_average=not train) if self.batchnorm else norm(x)

        return x, integration_timesteps
