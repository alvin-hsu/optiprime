import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from jax.random import PRNGKey, permutation
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from .rx_graph import RxGraph
from .rx_input import RxInput, start_worker_pool, stop_worker_pool


class RxDataset(object):
    def __init__(self, df: pd.DataFrame,
                 rate_plates: Sequence[str],
                 observables: Sequence[str],
                 json_path: Path,
                 csv_path: Optional[Path] = None,
                 rate_plate_sizes: Optional[Dict[str, int]] = None,
                 preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 rx_inputs: Optional[Sequence[RxInput]] = None,
                 multiprocess: bool = True,
                 verbose: bool = True):
        # Preprocess df if necessary
        self.df = df if preprocess_fn is None else preprocess_fn(csv_path, df)
        self.df = self.df.reset_index(drop=True)
        self.json_path = json_path
        self._cache = {'_META': {}}  # Input cache
        disk_cache = self.json_path.parent / '_disk_cache'
        self._cache['_META']['disk_cache'] = disk_cache
        # Load cached rate/observable data, if available
        if json_path.is_file():
            logging.debug('Found cached rate/observable information!')
            with json_path.open('r') as f:
                cached_data = json.load(f)
            self.rate_plates = cached_data['rate_plates']
            self.observables = cached_data['observables']
            self.rate_plate_sizes = cached_data['rate_plate_sizes']
            self.rate_plate_mapping = cached_data['rate_plate_mapping']
        else:
            self.rate_plates = rate_plates
            self.observables = observables
            self.rate_plate_sizes = rate_plate_sizes if rate_plate_sizes is not None else {}
            self.rate_plate_mapping = {}
            # Check that self.df has all the required columns
            for plate in self.rate_plates:
                assert plate in self.df.columns, f'Plate missing in dataframe: {plate}'
            for observed in observables:
                assert observed in self.df.columns, f'Observable missing in dataframe: {observed}'
            # Process plate index mappings
            for plate in self.rate_plates:
                column = self.df[plate]
                if plate in self.rate_plate_sizes:
                    self.rate_plate_mapping[plate] = list(range(rate_plate_sizes[plate]))
                    assert is_integer_dtype(column), f'Column {plate} must be an integer'
                    assert column.min() >= 0, f'Column {plate} has entry < 0'
                    assert column.max() < rate_plate_sizes[plate], \
                        f'Column {plate} has entry larger than {rate_plate_sizes[plate]}'
                else:
                    unique = column.unique().tolist()
                    self.rate_plate_mapping[plate] = unique
                    self.rate_plate_sizes[plate] = len(unique)
            logging.debug(f'Writing rate/observable information to {str(json_path)}')
            with json_path.open('w+') as f:
                json.dump({'rate_plates': self.rate_plates,
                           'observables': self.observables,
                           'rate_plate_sizes': self.rate_plate_sizes,
                           'rate_plate_mapping': self.rate_plate_mapping}, f)
        self.rate_plate_idxs = {}
        for plate in self.rate_plates:
            column = self.df[plate]
            mapping = self.rate_plate_mapping[plate]
            mapping = {x: i for i, x in enumerate(mapping)}
            self.rate_plate_idxs[plate] = column.apply(lambda x: mapping[x]).to_numpy()
        # Set observed
        self.observed = self.df[self.observables].to_numpy()
        # Rate inputs
        rx_inputs = {x: None for x in rx_inputs} if rx_inputs is not None else {}  # Using as set
        rx_inputs = [x for x in rx_inputs]
        self.rx_inputs = rx_inputs
        if rx_inputs is not None:
            if multiprocess:
                start_worker_pool()
            for rx_input in rx_inputs:
                logging.info(f'Preprocessing {rx_input.name}')
                rx_input.pre_process()
                rx_input.process(self.df, self._cache, verbose=verbose)
                rx_input.post_process()
            if multiprocess:
                stop_worker_pool()
        # Find groups
        self.groups = self.df['group'].unique().tolist()
        self.df['group_idx'] = self.df['group'].apply(lambda x: self.groups.index(x))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {rx_input.name: self.get_input(rx_input, idx) for rx_input in self.rx_inputs}

    def get_input(self, rx_input: RxInput, idx: int | slice | np.ndarray):
        return rx_input.get(self._cache, idx)

    def get_inputs(self, rx_inputs: Sequence[RxInput], idx: int | slice | np.ndarray):
        return [self.get_input(x, idx) for x in rx_inputs]

    def get_input_idxs(self, rx_inputs: Sequence[RxInput]):
        return [self.rx_inputs.index(x) for x in rx_inputs]

    def make_split(self, name: str, new_idxs: np.ndarray) -> 'RxDataset':
        logging.info(f'Creating {name} split with {len(new_idxs)} samples')
        new_idxs = np.sort(new_idxs)
        new_plate_idxs = {k: v[new_idxs] for k, v in self.rate_plate_idxs.items()}
        json_path = self.json_path.with_name(f'{name}_{self.json_path.name}')
        json_path.unlink(missing_ok=True)
        with json_path.open('w+') as f:
            json.dump({'rate_plates': self.rate_plates,
                       'observables': self.observables,
                       'rate_plate_sizes': self.rate_plate_sizes,
                       'rate_plate_mapping': self.rate_plate_mapping}, f)
        new_ds = RxDataset(df=self.df.iloc[new_idxs],
                           rate_plates=self.rate_plates,
                           observables=self.observables,
                           json_path=json_path)
        new_ds.rate_plate_idxs = new_plate_idxs
        new_ds.rate_plate_mapping = self.rate_plate_mapping
        new_ds.rx_inputs = self.rx_inputs
        new_ds._cache['_META']['disk_cache'] = self._cache['_META']['disk_cache']
        for rx_input in self.rx_inputs:
            rx_input.split(self._cache, new_ds._cache, new_idxs)
        return new_ds

    def split_preset(self, idx: Optional[int] = None, split_name: str = 'split'):
        if idx is None:
            return self, None
        # Train split
        train_idx = (self.df[split_name] != idx)
        train_idx = train_idx[train_idx].index.values
        train_ds = self.make_split(f'train_{idx}', train_idx)
        # Val split
        val_idx = (self.df[split_name] == idx)
        val_idx = val_idx[val_idx].index.values
        val_ds = self.make_split(f'val_{idx}', val_idx)
        return train_ds, val_ds

    def split_random(self, val_frac: float = 0.0, key: Optional[PRNGKey] = None):
        assert 0.0 <= val_frac <= 1.0, 'Validation fraction must be between 0 and 1'
        if val_frac == 0.0:
            return self, None
        n_val = int(val_frac * len(self))
        if key is not None:
            full_idx = permutation(key, len(self))
        else:
            logging.warning('Dataset splitting is untraceable! Consider specifying a random seed!')
            full_idx = np.random.permutation(len(self))
        train_idx, val_idx = full_idx[n_val:], full_idx[:n_val]
        train_ds = self.make_split('train', train_idx)
        val_ds = self.make_split('val', val_idx)
        return train_ds, val_ds

    @staticmethod
    def concatenate(datasets: Sequence['RxDataset'], json_path: Optional[Path] = None) \
            -> 'RxDataset':
        if len(datasets) == 1:
            return datasets[0]
        logging.info(f'Concatenating {len(datasets)} RxDatasets...')
        ds0 = datasets[0]
        new_df = pd.concat([ds.df for ds in datasets]).reset_index(drop=True)
        json_path = json_path if json_path is not None else Path('/tmp/concat.json')
        json_path.unlink(missing_ok=True)
        plate_mapping, plate_sizes, new_plate_idxs = {}, {}, {}
        for plate in ds0.rate_plates:
            # Get new plate mapping with unique values
            unique = set()
            for ds in datasets:
                unique.update(ds.rate_plate_mapping[plate])
            unique = list(unique)
            unique_map = {x: i for i, x in enumerate(unique)}
            plate_mapping[plate] = unique
            plate_sizes[plate] = len(unique)
            # Remap new dataset to new indices
            new_idxs = []
            for ds in datasets:
                ds_mapping = ds.rate_plate_mapping[plate]
                remap = [unique_map[x] for x in ds_mapping]
                old_idxs = ds.rate_plate_idxs[plate]
                new_idxs = new_idxs + [remap[x] for x in old_idxs]
            new_plate_idxs[plate] = np.array(new_idxs, dtype=np.uint32)
        with json_path.open('w+') as f:
            json.dump({'rate_plates': ds0.rate_plates,
                       'observables': ds0.observables,
                       'rate_plate_sizes': plate_sizes,
                       'rate_plate_mapping': plate_mapping}, f)
        new_ds = RxDataset(df=new_df,
                           rate_plates=ds0.rate_plates,
                           observables=ds0.observables,
                           json_path=json_path)
        for rx_input in ds0.rx_inputs:
            logging.debug(f'Concatenating {rx_input.name}...')
            rx_input.concat(new_ds._cache, [ds._cache for ds in datasets])
        new_ds.rx_inputs = ds0.rx_inputs
        new_ds._cache['_META']['disk_cache'] = ds0._cache['_META']['disk_cache']
        return new_ds

    @staticmethod
    def load_dir(path: Path,
                 rx_graph: RxGraph,
                 rx_inputs: Optional[Sequence[RxInput]] = None,
                 preprocess_fn: Optional[Callable[[Path, pd.DataFrame], pd.DataFrame]] = None):
        plate_names = rx_graph.plate_names
        obs_names = rx_graph.obs_names
        datasets = []
        for p in path.glob('*.csv'):
            df = pd.read_csv(p)
            json_path = p.resolve().with_suffix('.json')
            json_path.unlink(missing_ok=True)
            dataset = RxDataset(df=df,
                                rate_plates=plate_names,
                                observables=obs_names,
                                json_path=json_path,
                                csv_path=p,
                                preprocess_fn=preprocess_fn,
                                rx_inputs=rx_inputs)
            datasets.append(dataset)
        return RxDataset.concatenate(datasets, json_path=(path / 'concat.json'))
