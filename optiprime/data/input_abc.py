from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class SinglePool:
    """
    A dummy pool that performs work in series rather than parallel. Required if start_worker_pool
    and stop_worker_pool are not called.
    """
    def imap_unordered(self, func, iterable):
        for x in iterable:
            yield func(x)


_POOL: Pool = SinglePool()  # Set this before processing


def start_worker_pool():
    global _POOL
    n_workers = cpu_count() - 1
    logging.info(f'Starting worker pool with {n_workers=}')
    _POOL = Pool(n_workers)

def stop_worker_pool():
    global _POOL
    _POOL.close()  # noqa
    _POOL = None


class OpInput(ABC):
    name: str
    _instances = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__
        if cls.name not in OpInput._instances:
            OpInput._instances[cls.name] = cls()

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def decorator(cls):
        return OpInput._instances[cls.name]

    def pre_process(self):
        pass

    @abstractmethod
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        raise NotImplementedError('OpInputs must always implement process()')

    def post_process(self):
        pass

    def get(self, cache: Dict[str, Any], idx: int | slice | np.ndarray) -> np.ndarray:
        return cache[self.name][idx, ...]

    def split(self, full_cache: Dict[str, Any], sub_cache: Dict[str, Any],
              split_idx: int | slice | np.ndarray):
        sub_cache[self.name] = full_cache[self.name][split_idx, ...]

    def concat(self, full_cache: Dict[str, Any], sub_caches: List[Dict[str, Any]]):
        full_cache[self.name] = np.concatenate([sub_cache[self.name] for sub_cache in sub_caches])

    def tf_dataset(self, cache: Dict[str, Any]):
        # Only require tensorflow if training models
        import tensorflow as tf
        tf.config.set_visible_devices([], device_type='GPU')
        return tf.data.Dataset.from_tensor_slices(cache[self.name])

    def tf_process(self, x, cache):
        return x


class HashedScalarOpInput(OpInput, ABC):
    @abstractmethod
    def process_row(self, row: pd.Series) -> float:
        raise NotImplementedError('HashedScalarOpInputs must always implement .process_row()')

    @property
    @abstractmethod
    def hash_column(self):
        raise AttributeError('HashedOpInputs must set the .hash_column class variable')

    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        data = np.zeros(len(df), dtype=np.float32)
        seen_entries = {}
        for i, row in df.iterrows():
            if (hash_ := row[self.hash_column]) not in seen_entries:
                seen_entries[hash_] = self.process_row(row)
            data[i] = seen_entries[hash_]
        cache[self.name] = data


class HashedLargeOpInput(OpInput, ABC):
    _data_fmt = '_{}_DATA'
    _meta_fmt = '_{}_META'

    @property
    def data_name(self):
        return HashedLargeOpInput._data_fmt.format(self.name)

    @property
    def meta_name(self):
        return HashedLargeOpInput._meta_fmt.format(self.name)

    @abstractmethod
    def process_row(self, row: pd.Series) -> np.ndarray:
        raise NotImplementedError('HashedOpInputs must always implement .process_row()')

    @property
    @abstractmethod
    def hash_column(self):
        raise AttributeError('HashedOpInputs must set the .hash_column class variable')

    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = []
        cache[self.data_name] = {}
        cache[self.meta_name] = {}
        for i, row in df.iterrows():
            if (hash_ := row[self.hash_column]) not in cache[self.data_name]:
                data = self.process_row(row)
                cache[self.data_name][hash_] = data
            cache[self.name].append(hash_)

    def get_single(self, cache: Dict[str, Any], hash_: str):
        return cache[self.data_name][hash_]

    def get(self, cache: Dict[str, Any], idx: int | slice | np.ndarray) -> np.ndarray:
        # If shape is specified in metadata, then need to pad
        if 'shape' in cache[self.meta_name]:
            return self.get_padded(cache, idx)
        # Faster, but only works if entries are the same shape
        if isinstance(idx, int):
            return self.get_single(cache, cache[self.name][idx])
        else:  # slice or np.ndarray
            hashes = cache[self.name][idx] if isinstance(idx, slice) else \
                     [cache[self.name][i] for i in idx]
            return np.stack([self.get_single(cache, hash_) for hash_ in hashes], axis=0)

    def get_padded(self, cache: Dict[str, Any], idx: int | slice | np.ndarray) -> np.ndarray:
        shape = cache[self.meta_name]['shape']
        if isinstance(idx, int):
            retval = np.zeros(shape, dtype=np.float32)
            data = self.get_single(cache, cache[self.name][idx])
            r_slice = tuple([slice(0, x) for x in data.shape])
            retval[r_slice] = data
            return retval
        else:  # slice or np.ndarray
            hashes = cache[self.name][idx] if isinstance(idx, slice) else \
                     [cache[self.name][i] for i in idx]
            retval = np.zeros((len(hashes),) + shape, dtype=np.float32)
            for i, hash_ in enumerate(hashes):
                data = self.get_single(cache, hash_)
                r_slice = tuple([i] + [slice(0, x) for x in data.shape])
                retval[r_slice] = data
            return retval

    def split(self, full_cache: Dict[str, Any], sub_cache: Dict[str, Any],
              split_idx: slice | np.ndarray):
        sub_cache[self.name] = full_cache[self.name][split_idx] if isinstance(split_idx, slice) \
                               else [full_cache[self.name][i] for i in split_idx]
        sub_cache[self.data_name] = {k: full_cache[self.data_name][k]
                                     for k in sub_cache[self.name]}
        sub_cache[self.meta_name] = deepcopy(full_cache[self.meta_name])

    def concat(self, full_cache: Dict[str, Any], sub_caches: List[Dict[str, Any]]):
        full_cache[self.name] = sum([sub_cache[self.name] for sub_cache in sub_caches], [])
        full_cache[self.data_name] = {}
        for sub_cache in sub_caches:
            full_cache[self.data_name].update(sub_cache[self.data_name])
        full_cache[self.meta_name] = deepcopy(sub_caches[0][self.meta_name])

    def tf_dataset(self, cache: Dict[str, Any]):
        # Only require tensorflow if training models
        import tensorflow as tf
        tf.config.set_visible_devices([], device_type='GPU')
        return tf.data.Dataset.from_tensor_slices(self.get(cache, slice(None)))
