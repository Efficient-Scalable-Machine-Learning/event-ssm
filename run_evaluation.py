import hydra
from omegaconf import OmegaConf as om
from omegaconf import DictConfig, open_dict
from functools import partial

import jax.random
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from event_ssm.dataloading import Datasets
from event_ssm.ssm import init_S5SSM
from event_ssm.seq_model import BatchClassificationModel


def setup_evaluation(cfg: DictConfig):
    num_devices = jax.local_device_count()
    assert cfg.checkpoint, "No checkpoint directory provided. Use checkpoint=<path> to specify a checkpoint."

    # load task specific data
    create_dataset_fn = Datasets[cfg.task.name]

    # Create dataset...
    print("[*] Loading dataset...")
    train_loader, val_loader, test_loader, data = create_dataset_fn(
        cache_dir=cfg.data_dir,
        seed=cfg.seed,
        world_size=num_devices,
        **cfg.training
    )

    with open_dict(cfg):
        # optax updates the schedule every iteration and not every epoch
        cfg.optimizer.total_steps = cfg.training.num_epochs * len(train_loader) // cfg.optimizer.accumulation_steps
        cfg.optimizer.warmup_steps = cfg.optimizer.warmup_epochs * len(train_loader) // cfg.optimizer.accumulation_steps

        # scale learning rate by batch size
        cfg.optimizer.ssm_lr = cfg.optimizer.ssm_base_lr * cfg.training.per_device_batch_size * num_devices * cfg.optimizer.accumulation_steps

    # load model
    print("[*] Creating model...")
    ssm_init_fn = init_S5SSM(**cfg.model.ssm_init)
    model = BatchClassificationModel(
        ssm=ssm_init_fn,
        num_classes=data.n_classes,
        num_embeddings=data.num_embeddings,
        **cfg.model.ssm,
    )

    # initialize training state
    state = checkpoints.restore_checkpoint(cfg.checkpoint, target=None)
    params = state['params']
    model_state = state['model_state']

    return model, params, model_state, train_loader, val_loader, test_loader


def evaluation_step(
        apply_fn,
        params,
        model_state,
        batch
):
    """
    Evaluates the loss of the function passed as argument on a batch

    :param train_state: a Flax TrainState that carries the parameters, optimizer states etc
    :param batch: the data consisting of [data, target]
    :return: train_state, metrics
    """
    inputs, targets, integration_timesteps, lengths = batch
    logits = apply_fn(

        {'params': params, **model_state},
        inputs, integration_timesteps, lengths,
        False,
    )

    loss = optax.softmax_cross_entropy(logits, targets)
    loss = loss.mean()
    preds = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(targets, axis=-1)
    accuracy = (preds == targets).mean()

    return {'loss': loss, 'accuracy': accuracy}, preds


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(config: DictConfig):
    print(om.to_yaml(config))

    model, params, model_state, train_loader, val_loader, test_loader = setup_evaluation(cfg=config)
    step = partial(evaluation_step, model.apply, params, model_state)
    step = jax.jit(step)

    # run training
    print("[*] Running evaluation...")
    metrics = {}
    events_per_sample = []
    time_per_sample = []
    targets = []
    predictions = []
    num_batches = 0

    for i, batch in enumerate(test_loader):
        step_metrics, preds = step(batch)

        predictions.append(preds)
        targets.append(jnp.argmax(batch[1], axis=-1))
        time_per_sample.append(jnp.sum(batch[2], axis=1))
        events_per_sample.append(batch[3])

        if not metrics:
            metrics = step_metrics
        else:
            for key, val in step_metrics.items():
                metrics[key] += val
        num_batches += 1

    metrics = {key: jnp.mean(metrics[key] / num_batches).item() for key in metrics}

    print(f"[*] Test accuracy: {100 * metrics['accuracy']:.2f}%")


if __name__ == '__main__':
    main()
