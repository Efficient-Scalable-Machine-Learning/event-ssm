import numpy as np


class Identity:
    def __call__(self, events):
        return events


class CropEvents:
    """Crops event stream to a specified number of events

    Parameters:
        num_events (int): number of events to keep
    """

    def __init__(self, num_events):
        self.num_events = num_events

    def __call__(self, events):
        if self.num_events >= len(events):
            return events
        else:
            start = np.random.randint(0, len(events) - self.num_events)
            return events[start:start + self.num_events]


class Jitter1D:
    """
    Apply random jitter to event coordinates
    Parameters:
        max_roll (int): maximum number of pixels to roll by
    """
    def __init__(self, sensor_size, var):
        self.sensor_size = sensor_size
        self.var = var

    def __call__(self, events):
        # roll x, y coordinates by a random amount
        shift = np.random.normal(0, self.var, len(events)).astype(np.int32)
        events['x'] += shift
        # remove events who got shifted out of the sensor size
        mask = (events['x'] >= 0) & (events['x'] < self.sensor_size[0])
        events = events[mask]
        return events


class Roll:
    """
    Roll event x, y coordinates by a random amount

    Parameters:
        max_roll (int): maximum number of pixels to roll by
    """
    def __init__(self, sensor_size, p, max_roll):
        self.sensor_size = sensor_size
        self.max_roll = max_roll
        self.p = p

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # roll x, y coordinates by a random amount
        roll_x = np.random.randint(-self.max_roll, self.max_roll)
        roll_y = np.random.randint(-self.max_roll, self.max_roll)
        events['x'] += roll_x
        events['y'] += roll_y
        # remove events who got shifted out of the sensor size
        mask = (events['x'] >= 0) & (events['x'] < self.sensor_size[0]) & (events['y'] >= 0) & (events['y'] < self.sensor_size[1])
        events = events[mask]
        return events


class Rotate:
    """
    Rotate event x, y coordinates by a random angle
    """
    def __init__(self, sensor_size, p, max_angle):
        self.p = p
        self.sensor_size = sensor_size
        self.max_angle = 2 * np.pi * max_angle / 360

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # rotate x, y coordinates by a random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        x = events['x'] - self.sensor_size[0] / 2
        y = events['y'] - self.sensor_size[1] / 2
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        events['x'] = (x_new + self.sensor_size[0] / 2).astype(np.int32)
        events['y'] = (y_new + self.sensor_size[1] / 2).astype(np.int32)
        # clip to original range
        events['x'] = np.clip(events['x'], 0, self.sensor_size[0])
        events['y'] = np.clip(events['y'], 0, self.sensor_size[1])
        return events


class Scale:
    """
    Scale event x, y coordinates by a random factor
    """
    def __init__(self, sensor_size, p, max_scale):
        assert max_scale >= 1
        self.p = p
        self.sensor_size = sensor_size
        self.max_scale = max_scale

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        # scale x, y coordinates by a random factor
        scale = np.random.uniform(1/self.max_scale, self.max_scale)
        x = events['x'] - self.sensor_size[0] / 2
        y = events['y'] - self.sensor_size[1] / 2
        x_new = x * scale
        y_new = y * scale
        events['x'] = (x_new + self.sensor_size[0] / 2).astype(np.int32)
        events['y'] = (y_new + self.sensor_size[1] / 2).astype(np.int32)
        # remove events who got shifted out of the sensor size
        mask = (events['x'] >= 0) & (events['x'] < self.sensor_size[0]) & (events['y'] >= 0) & (events['y'] < self.sensor_size[1])
        events = events[mask]
        return events


class DropEventChunk:
    """
    Randomly drop a chunk of events
    """
    def __init__(self, p, max_drop_size):
        self.drop_prob = p
        self.max_drop_size = max_drop_size

    def __call__(self, events):
        max_drop_events = self.max_drop_size * len(events)
        if np.random.rand() < self.drop_prob:
            drop_size = np.random.randint(1, max_drop_events)
            start = np.random.randint(0, len(events) - drop_size)
            events = np.delete(events, slice(start, start + drop_size), axis=0)
        return events


class OneHotLabels:
    """
    Convert integer labels to one-hot encoding
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        return np.eye(self.num_classes)[label]


def cut_mix_augmentation(events, targets):
    """
    Cut and mix two event streams by a random event chunk. Input is a list of event streams.

    Args:
        events (dict): batch of event streams of shape (batch_size, num_events, 4)
        max_num_events (int): maximum number of events to mix
    """
    # get the total time of all events
    lengths = np.array([e.shape[0] for e in events])

    # get fraction of the event-stream to cut
    cut_size = np.random.randint(low=1, high=lengths)
    start_event = np.random.randint(low=0, high=lengths - cut_size)

    # a random permutation to mix the events
    rand_index = np.random.permutation(len(events))

    mixed_events = []
    mixed_targets = []

    # cut events from b and mix them with events from a
    for i in range(len(events)):
        events_b = events[rand_index[i]][start_event[rand_index[i]]:start_event[rand_index[i]] + cut_size[rand_index[i]]]
        mask_a = (events[i]['t'] >= events_b['t'][0]) & (events[i]['t'] <= events_b['t'][-1])
        events_a = events[i][~mask_a]

        # mix and sort events
        new_events = np.concatenate([events_a, events_b])
        new_events = new_events[np.argsort(new_events['t'])]

        # mix targets
        lam = events_b.shape[0] / new_events.shape[0]
        assert 0 <= lam <= 1, f'lam should be between 0 and 1, but got {lam} {cut_size[rand_index[i]]} {events_a.shape[0]} {events_b.shape[0]}'

        # append mixed events and targets
        mixed_events.append(new_events)
        mixed_targets.append(targets[i] * (1 - lam) + targets[rand_index[i]] * lam)

    return mixed_events, mixed_targets


def cut_mix_augmentation_time(events, targets):
    """
    Cut and mix two event streams by a random event chunk. Input is a list of event streams.

    :param events: batch of event streams of shape (batch_size, num_events, 4)
    :param targets: batch of targets of shape (batch_size, num_classes)

    :return: mixed events, mixed targets
    """
    # get the total time of all events
    lengths = np.array([e['t'][-1] - e['t'][0] for e in events], dtype=np.float32)

    # get fraction of the event-stream to cut
    cut_size = np.random.uniform(low=0, high=lengths)
    start_time = np.random.uniform(low=0, high=lengths - cut_size)

    # a random permutation to mix the events
    rand_index = np.random.permutation(len(events))

    mixed_events = []
    mixed_targets = []

    # cut events from b and mix them with events from a
    for i in range(len(events)):
        start, end = start_time[rand_index[i]], start_time[rand_index[i]] + cut_size[rand_index[i]]
        mask_a = (events[i]['t'] >= start) & (events[i]['t'] <= end)
        mask_b = (events[rand_index[i]]['t'] >= start) & (events[rand_index[i]]['t'] <= end)

        # mix events
        new_events = np.concatenate([events[i][~mask_a], events[rand_index[i]][mask_b]])

        # avoid the case that the new events are empty
        if len(new_events) == 0:
            mixed_events.append(events[i])
            mixed_targets.append(targets[i])
        else:
            # sort events
            new_events = new_events[np.argsort(new_events['t'])]
            mixed_events.append(new_events)

            # mix targets
            new_length = new_events['t'][-1] - new_events['t'][0]
            if len(events[rand_index[i]]['t'][mask_b]) == 0:
                cut_length = 0
            else:
                cut_length = events[rand_index[i]]['t'][mask_b][-1] - events[rand_index[i]]['t'][mask_b][0]
            lam = cut_length / new_length
            assert 0 <= lam <= 1, f'lam should be between 0 and 1, but got {lam} {new_length} {cut_size[rand_index[i]]} {start} {end}'
            mixed_targets.append(targets[i] * (1 - lam) + targets[rand_index[i]] * lam)

    return mixed_events, mixed_targets
