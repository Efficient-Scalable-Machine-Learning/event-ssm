from functools import partial
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal, glorot_normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, make_DPLR_HiPPO

from .layers import EventPooling


def discretize_zoh(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    This is the default discretization method used by many SSM works including S5.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = (1/Lambda * (Lambda_bar-Identity))
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes.
    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Delta = step_delta * time_delta
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = 1.0
    return Lambda_bar, gamma_bar


def discretize_async(Lambda, step_delta, time_delta):
    """
    Discretize a diagonalized, continuous-time linear SSM
    with dirac delta input spikes and appropriate input normalization.

    :param Lambda: diagonal state matrix (P,)
    :param step_delta: discretization step sizes (P,)
    :param time_delta: (float32) discretization step sizes (P,)
    :return: discretized Lambda_bar (complex64), B_bar (complex64) (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * step_delta * time_delta)
    gamma_bar = (1/Lambda * (np.exp(Lambda * step_delta)-Identity))
    return Lambda_bar, gamma_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """
    Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    :param q_i: tuple containing A_i and Bu_i at position i (P,), (P,)
    :param q_j: tuple containing A_j and Bu_j at position j (P,), (P,)
    :return: new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, stride=1):
    """
    Compute the LxH output of discretized SSM given an LxH input.

    :param Lambda_elements: (complex64) discretized state matrix (L, P)
    :param Bu_elements: (complex64) discretized inputs projected to state space (L, P)
    :param C_tilde: (complex64) output matrix (H, P)
    :param conj_sym: (bool) whether conjugate symmetry is enforced
    :return: ys: (float32) the SSM outputs (S5 layer preactivations) (L, H)
    """
    remaining_timesteps = (Bu_elements.shape[0] // stride) * stride

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    xs = xs[:remaining_timesteps:stride]

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5SSM(nn.Module):
    H_in: int
    H_out: int
    P: int
    block_size: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    stride: int = 1
    pooling_mode: str = "last"

    """
    Event-based S5 module
    
    :param H_in: int, SSM input dimension
    :param H_out: int, SSM output dimension
    :param P: int, SSM state dimension
    :param block_size: int, block size for block-diagonal state matrix
    :param C_init: str, initialization method for output matrix C
    :param discretization: str, discretization method for event-based SSM
    :param dt_min: float, minimum value of log timestep
    :param dt_max: float, maximum value of log timestep
    :param conj_sym: bool, whether to enforce conjugate symmetry in the state space operator
    :param clip_eigs: bool, whether to clip eigenvalues of the state space operator
    :param step_rescale: float, rescale factor for step size
    :param stride: int, stride for subsampling layer
    :param pooling_mode: str, pooling mode for subsampling layer
    """

    def setup(self):
        """
        Initializes parameters once and performs discretization each time the SSM is applied to a sequence
        """

        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(self.block_size)

        blocks = self.P // self.block_size
        block_size = self.block_size // 2 if self.conj_sym else self.block_size
        local_P = self.P // 2 if self.conj_sym else self.P

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        state_str = f"SSM: {self.H_in} -> {self.P} -> {self.H_out}"
        if self.stride > 1:
            state_str += f" (stride {self.stride} with pooling mode {self.pooling_mode})"
        print(state_str)

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (None,))

        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (self.P, self.H_in)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init, rng, shape, Vinv),
                            B_shape)

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H_out, self.P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H_out, self.P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            C = self.param("C", C_init, (self.H_out, local_P, 2))
            self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init, rng, shape, V),
                                C_shape)

            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        if self.H_in == self.H_out:
            self.D = self.param("D", normal(stddev=1.0), (self.H_in,))
        else:
            self.D = self.param("D", glorot_normal(), (self.H_out, self.H_in))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (local_P, self.dt_min, self.dt_max))

        # pooling layer
        self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

        # Discretize
        if self.discretization in ["zoh"]:
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["dirac"]:
            self.discretize_fn = discretize_dirac
        elif self.discretization in ["async"]:
            self.discretize_fn = discretize_async
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence, integration_timesteps):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence using a parallel scan.

        :param input_sequence: (float32) input sequence (L, H)
        :param integration_timesteps: (float32) integration timesteps (L,)
        :return: (float32) output sequence (L, H)
        """

        # discretize on the fly
        B = self.B[..., 0] + 1j * self.B[..., 1]

        def discretize_and_project_inputs(u, _timestep):
            step = self.step_rescale * np.exp(self.log_step[:, 0])
            Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step, _timestep)
            Bu = gamma_bar * (B @ u)
            return Lambda_bar, Bu

        Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(input_sequence, integration_timesteps)

        ys = apply_ssm(
            Lambda_bar_elements,
            Bu_bar_elements,
            self.C_tilde,
            self.conj_sym,
            stride=self.stride
        )

        if self.stride > 1:
            input_sequence, _ = self.pool(input_sequence, integration_timesteps)

        if self.H_in == self.H_out:
            Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        else:
            Du = jax.vmap(lambda u: self.D @ u)(input_sequence)

        return ys + Du


def init_S5SSM(
        C_init,
        dt_min,
        dt_max,
        conj_sym,
        clip_eigs,
):
    """
    Convenience function that will be used to initialize the SSM.
    Same arguments as defined in S5SSM above.
    """
    return partial(S5SSM,
                   C_init=C_init,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs
                   )
