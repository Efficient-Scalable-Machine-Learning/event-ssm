import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
import numpy as np
from event_ssm.transform import Identity, Roll, Rotate, Scale, DropEventChunk, Jitter1D, OneHotLabels, cut_mix_augmentation

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]


class Data:
    """
    Data class for storing dataset specific information
    """
    def __init__(
            self,
            n_classes: int,
            num_embeddings: int,
            train_size: int
):
        self.n_classes = n_classes
        self.num_embeddings = num_embeddings
        self.train_size = train_size


def event_stream_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False):
    """
    Collate function to turn event stream data into tokens ready for the JAX model

    :param batch: list of tuples of (events, target)
    :param resolution: resolution of the event stream
    :param pad_unit: padding unit for the tokens. All sequences will be padded to integer multiples of this unit.
                     This option results in JAX compiling multiple GPU kernels for different sequence lengths,
                     which might slow down compilation time, but improves throughput for the rest of the training process.
    :param cut_mix: probability of applying cut mix augmentation
    :param no_time_information: if True, the time information is ignored and all events are treated as if they were
                                recorded sampled at uniform time intervals.
                                This option is only used for ablation studies.
    """
    # x are inputs, y are targets, z are aux data
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    # apply cut mix augmentation
    if np.random.rand() < cut_mix:
        x, y = cut_mix_augmentation(x, y)

    # set labels to numpy array
    y = np.stack(y)

    # integration time steps are the difference between two consequtive time stamps
    if no_time_information:
        timesteps = [np.ones_like(e['t'][:-1]) for e in x]
    else:
        timesteps = [np.diff(e['t']) for e in x]

    # NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

    # process tokens for single input dim (e.g. audio)
    if len(resolution) == 1:
        tokens = [e['x'][:-1].astype(np.int32) for e in x]
    elif len(resolution) == 2:
        tokens = [(e['x'][:-1] * e['y'][:-1] + np.prod(resolution) * e['p'][:-1].astype(np.int32)).astype(np.int32) for e in x]
    else:
        raise ValueError('resolution must contain 1 or 2 elements')

    # get padding lengths
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

    # pad tokens with -1, which results in a zero vector with embedding look-ups
    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    # timesteps are in micro seconds... transform to milliseconds
    timesteps = timesteps / 1000

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def event_stream_dataloader(
        train_data,
        val_data,
        test_data,
        batch_size,
        eval_batch_size,
        train_collate_fn,
        eval_collate_fn,
        rng,
        num_workers=0,
        shuffle_training=True
):
    """
    Create dataloaders for training, validation and testing

    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: test dataset
    :param batch_size: batch size for training
    :param eval_batch_size: batch size for evaluation
    :param train_collate_fn: collate function for training
    :param eval_collate_fn: collate function for evaluation
    :param rng: random number generator
    :param num_workers: number of workers for data loading
    :param shuffle_training: whether to shuffle the training data

    :return: train_loader, val_loader, test_loader
    """
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=bsz,
            drop_last=drop_last,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=rng,
            num_workers=num_workers
        )
    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def create_events_shd_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        validate_on_test: bool = False,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    creates a view of the spiking heidelberg digits dataset

    :param cache_dir:		    (str):		where to store the dataset
    :param bsz:				    (int):		Batch size.
    :param seed:			    (int)		Seed for shuffling data.
    :param time_jitter:		    (float)		Standard deviation of the time jitter.
    :param spatial_jitter:	    (float)		Standard deviation of the spatial jitter.
    :param max_drop_chunk:	    (float)		Maximum fraction of events to drop in a single chunk.
    :param noise:			    (int)		Number of noise events to add.
    :param drop_event:		    (float)		Probability of dropping an event.
    :param time_skew:		    (float)		Time skew factor.
    :param cut_mix:			    (float)		Probability of applying cut mix augmentation.
    :param pad_unit:		    (int)		Padding unit for the tokens. See collate function for more details
    :param validate_on_test:	(bool)		If True, use the test set for validation.
                                            Else use a random validation split from the test set.
    :param no_time_information:	(bool)		Whether to ignore the time information in the events.

    :return: train_loader, val_loader, test_loader, data
    """
    print("[*] Generating Spiking Heidelberg Digits Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    sensor_size = (700, 1, 1)

    transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ])
    target_transforms = OneHotLabels(num_classes=20)

    train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SHD(save_to=cache_dir, train=True, target_transform=target_transforms)
    test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)

    # create validation set
    if validate_on_test:
        print("[*] WARNING: Using test set for validation")
        val_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)
    else:
        val_length = int(0.1 * len(train_data))
        indices = torch.randperm(len(train_data), generator=rng)
        train_data = torch.utils.data.Subset(train_data, indices[:-val_length])
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])

    collate_fn = partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    data = Data(
        n_classes=20, num_embeddings=700, train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_ssc_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    creates a view of the spiking speech commands dataset

    :param cache_dir:		    (str):		where to store the dataset
    :param bsz:				    (int):		Batch size.
    :param seed:			    (int)		Seed for shuffling data.
    :param time_jitter:		    (float)		Standard deviation of the time jitter.
    :param spatial_jitter:	    (float)		Standard deviation of the spatial jitter.
    :param max_drop_chunk:	    (float)		Maximum fraction of events to drop in a single chunk.
    :param noise:			    (int)		Number of noise events to add.
    :param drop_event:		    (float)		Probability of dropping an event.
    :param time_skew:		    (float)		Time skew factor.
    :param cut_mix:			    (float)		Probability of applying cut mix augmentation.
    :param pad_unit:		    (int)		Padding unit for the tokens. See collate function for more details
    :param no_time_information:	(bool)		Whether to ignore the time information in the events.

    :return: train_loader, val_loader, test_loader, data
    """
    print("[*] Generating Spiking Speech Commands Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    sensor_size = (700, 1, 1)

    transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ])
    target_transforms = OneHotLabels(num_classes=35)

    train_data = tonic.datasets.SSC(save_to=cache_dir, split='train', transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid', target_transform=target_transforms)
    test_data = tonic.datasets.SSC(save_to=cache_dir, split='test', target_transform=target_transforms)

    collate_fn = partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=35, num_embeddings=700, train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_dvs_gesture_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        slice_events: int = 0,
        pad_unit: int = 2 ** 19,
        # Augmentation parameters
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        downsampling: int = 1,
        max_roll: int = 4,
        max_angle: float = 10,
        max_scale: float = 1.5,
        max_drop_chunk: float = 0.1,
        validate_on_test: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    creates a view of the DVS Gesture dataset

    :param cache_dir:		    (str):		where to store the dataset
    :param bsz:				    (int):		Batch size.
    :param seed:			    (int)		Seed for shuffling data.
    :param slice_events:	    (int)		Number of events per slice.
    :param pad_unit:		    (int)		Padding unit for the tokens. See collate function for more details
    :param time_jitter:		    (float)		Standard deviation of the time jitter.
    :param spatial_jitter:	    (float)		Standard deviation of the spatial jitter.
    :param noise:			    (int)		Number of noise events to add.
    :param drop_event:		    (float)		Probability of dropping an event.
    :param time_skew:		    (float)		Time skew factor.
    :param cut_mix:			    (float)		Probability of applying cut mix augmentation.
    :param downsampling:	    (int)		Downsampling factor.
    :param max_roll:		    (int)		Maximum number of pixels to roll the events.
    :param max_angle:		    (float)		Maximum angle to rotate the events.
    :param max_scale:		    (float)		Maximum scale factor to scale the events.
    :param max_drop_chunk:	    (float)		Maximum fraction of events to drop in a single chunk.
    :param validate_on_test:	(bool)		If True, use the test set for validation.
                                            Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """
    print("[*] Generating DVS Gesture Classification Dataset")

    assert time_skew > 1, "time_skew must be greater than 1"

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    orig_sensor_size = (128, 128, 2)
    new_sensor_size = (128 // downsampling, 128 // downsampling, 2)
    train_transforms = [
        # Event transformations
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        tonic.transforms.DropEvent(p=drop_event),
        tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise)),
        # Time tranformations
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        # Spatial transformations
        tonic.transforms.SpatialJitter(sensor_size=orig_sensor_size, var_x=spatial_jitter, var_y=spatial_jitter, clip_outliers=True),
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
        # Geometric tranformations
        Roll(sensor_size=new_sensor_size, p=0.3, max_roll=max_roll),
        Rotate(sensor_size=new_sensor_size, p=0.3, max_angle=max_angle),
        Scale(sensor_size=new_sensor_size, p=0.3, max_scale=max_scale),
    ]

    train_transforms = tonic.transforms.Compose(train_transforms)
    test_transforms = tonic.transforms.Compose([
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
    ])
    target_transforms = OneHotLabels(num_classes=11)

    TrainData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=True)
    TestData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=False)

    # create validation set
    if validate_on_test:
        print("[*] WARNING: Using test set for validation")
        val_data = TestData(transform=test_transforms, target_transform=target_transforms)
    else:
        # create train validation split
        val_data = TrainData(transform=test_transforms, target_transform=target_transforms)
        val_length = int(0.2 * len(val_data))
        indices = torch.randperm(len(val_data), generator=rng)
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])

    # if slice event count is given, train on slices of the training data
    if slice_events > 0:
        slicer = tonic.slicers.SliceByEventCount(event_count=slice_events, overlap=slice_events // 2, include_incomplete=True)
        train_subset = torch.utils.data.Subset(TrainData(), indices[:-val_length]) if not validate_on_test else TrainData()
        train_data = tonic.sliced_dataset.SlicedDataset(
            dataset=train_subset,
            slicer=slicer,
            transform=train_transforms,
            target_transform=target_transforms,
            metadata_path=None
        )
    else:
        train_data = torch.utils.data.Subset(
            TrainData(transform=train_transforms, target_transform=target_transforms),
            indices[:-val_length]
        ) if not validate_on_test else TrainData(transform=train_transforms)

    # Always evaluate on the full sequences
    test_data = TestData(transform=test_transforms, target_transform=target_transforms)

    # define collate functions
    train_collate_fn = partial(
            event_stream_collate_fn,
            resolution=new_sensor_size[:2],
            pad_unit=slice_events if (slice_events != 0 and slice_events < pad_unit) else pad_unit,
            cut_mix=cut_mix
        )
    eval_collate_fn = partial(
            event_stream_collate_fn,
            resolution=new_sensor_size[:2],
            pad_unit=pad_unit,
        )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=train_collate_fn,
        eval_collate_fn=eval_collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=11, num_embeddings=np.prod(new_sensor_size), train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


Datasets = {
    "shd-classification": create_events_shd_classification_dataset,
    "ssc-classification": create_events_ssc_classification_dataset,
    "dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
}
