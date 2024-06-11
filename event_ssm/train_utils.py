import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Any, Dict
import random
from flax.training import train_state
import optax
from functools import partial


class TrainState(train_state.TrainState):
    key: Array
    model_state: Dict


def training_step(
        train_state: TrainState,
        batch: Array,
        dropout_key: Array,
        distributed: bool = False
):
    """
    Conducts a single training step on a batch of data.

    :param train_state: a Flax TrainState that carries the parameters, optimizer states etc
    :param batch: the data consisting of [data, target, integration_timesteps, lengths]
    :param distributed: If True, apply reduce operations like psum, pmean etc
    :return: train_state, metrics
    """
    inputs, targets, integration_timesteps, lengths = batch

    def loss_fn(params):
        logits, updates = train_state.apply_fn(
            {'params': params, **train_state.model_state},
            inputs, integration_timesteps, lengths,
            True,
            rngs={'dropout': dropout_key},
            mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy(logits, targets)
        loss = loss.mean()

        return loss, (logits, updates)

    (loss, (logits, batch_updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)

    preds = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(targets, axis=-1)
    accuracy = (preds == targets).mean()

    if distributed:
        grads = jax.lax.pmean(grads, axis_name='data')
        loss = jax.lax.pmean(loss, axis_name='data')
        accuracy = jax.lax.pmean(accuracy, axis_name='data')

    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(model_state=batch_updates)

    return train_state, {'loss': loss, 'accuracy': accuracy}


def evaluation_step(
        train_state: TrainState,
        batch: Array,
        distributed: bool = False
):
    """
    Conducts a single evaluation step on a batch of data.

    :param train_state: a Flax TrainState that carries the parameters, optimizer states etc
    :param batch: the data consisting of [data, target]
    :param distributed: If True, apply reduce operations like psum, pmean etc
    :return: train_state, metrics
    """
    inputs, targets, integration_timesteps, lengths = batch
    logits = train_state.apply_fn(
        {'params': train_state.params, **train_state.model_state},
        inputs, integration_timesteps, lengths,
        False,
    )
    loss = optax.softmax_cross_entropy(logits, targets)
    loss = loss.mean()
    preds = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(targets, axis=-1)
    accuracy = (preds == targets).mean()

    if distributed:
        loss = jax.lax.pmean(loss, axis_name='data')
        accuracy = jax.lax.pmean(accuracy, axis_name='data')

    return train_state, {'loss': loss, 'accuracy': accuracy}


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def map_nested_fn_with_keyword(keyword_1, keyword_2):
    '''labels all the leaves that are descendants of keyword_1 with keyword 1,
    else label the leaf with keyword_2'''

    def map_fn(nested_dict):
        output_dict = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                if k == keyword_1:
                    output_dict[k] = map_fn_2(v)
                else:
                    output_dict[k] = map_fn(v)
            else:
                if k == keyword_1:
                    output_dict[k] = keyword_1
                else:
                    output_dict[k] = keyword_2
        return output_dict

    def map_fn_2(nested_dict):
        output_dict = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                output_dict[k] = map_fn_2(v)
            else:
                output_dict[k] = keyword_1
        return output_dict

    return map_fn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_first_device(x):
    x = jax.tree_util.tree_map(lambda a: a[0], x)
    return jax.device_get(x)


def print_model_size(params, name=''):
    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    total_params_size = sum(jax.tree_leaves(param_sizes))
    print('[*] Model parameter count:', total_params_size)


def get_learning_rate_fn(lr, total_steps, warmup_steps, schedule, **kwargs):
    if schedule == 'cosine':
        learning_rate_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps
        )
    elif schedule == 'constant':
        learning_rate_fn = optax.join_schedules([
            optax.linear_schedule(
                init_value=0.,
                end_value=lr,
                transition_steps=warmup_steps
            ),
            optax.constant_schedule(lr)
        ], [warmup_steps])
    else:
        raise ValueError(f'Unknown schedule: {schedule}')

    return learning_rate_fn


def get_optimizer(opt_config):

    ssm_lrs = ["B", "Lambda_re", "Lambda_im"]
    ssm_fn = map_nested_fn(
        lambda k, _: "ssm"
        if k in ssm_lrs
        else "regular"
    )
    learning_rate_fn = partial(
        get_learning_rate_fn,
        total_steps=opt_config.total_steps,
        warmup_steps=opt_config.warmup_steps,
        schedule=opt_config.schedule
    )

    def optimizer(learning_rate):
        tx = optax.multi_transform(
            {
                "ssm": optax.inject_hyperparams(partial(
                    optax.adamw,
                    b1=0.9, b2=0.999,
                    weight_decay=opt_config.ssm_weight_decay
                ))(learning_rate=learning_rate_fn(lr=learning_rate)),
                "regular": optax.adamw(
                    learning_rate=learning_rate_fn(lr=learning_rate * opt_config.lr_factor),
                    b1=0.9, b2=0.999,
                    weight_decay=opt_config.weight_decay),
            },
            ssm_fn,
        )
        if opt_config.get('accumulation_steps', False):
            print(f"[*] Using gradient accumulation with {opt_config.accumulation_steps} steps")
            tx = optax.MultiSteps(tx, every_k_schedule=opt_config.accumulation_steps)
        return tx

    return optimizer(opt_config.ssm_lr)


def init_model_state(rng_key, model, inputs, steps, lengths, opt_config):
    """
    Initialize the training state.

    :param rng_key: a PRNGKey
    :param model: the Flax model to train
    :param inputs: dummy input data
    :param steps: dummy integration timesteps
    :param lengths: dummy number of events
    :param opt_config: a dictionary containing the optimizer configuration
    :return: a TrainState object
    """
    init_key, dropout_key = jax.random.split(rng_key)
    variables = model.init(
        {"params": init_key,
         "dropout": dropout_key},
        inputs, steps, lengths, True
    )
    params = variables.pop('params')
    model_state = variables
    print_model_size(params)

    tx = get_optimizer(opt_config)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        key=dropout_key,
        model_state=model_state
    )
