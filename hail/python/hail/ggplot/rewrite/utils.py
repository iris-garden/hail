from dataclasses import asdict, dataclass


frozen_dataclass = dataclass(frozen=True)


def as_nonempty_dict(data):
    return asdict(data, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


def merge(data1, data2):
    return data1.__class__({**as_nonempty_dict(data1), **as_nonempty_dict(data2)})
