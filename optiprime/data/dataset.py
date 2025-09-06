import json
import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

from jax.random import PRNGKey, permutation
import numpy as np
import pandas as pd

from .input_abc import OpInput, start_worker_pool, stop_worker_pool


class OpDataset(object):
    def __init__(self, df: pd.DataFrame,
                 observables: Sequence[str],
                 json_path: Path,
                 csv_path: Optional[Path] = None,
                 preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 rx_inputs: Optional[Sequence[OpInput]] = None,
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
            self.observables = cached_data['observables']
        else:
            self.observables = observables
            # Check that self.df has all the required columns
            for observed in observables:
                assert observed in self.df.columns, f'Observable missing in dataframe: {observed}'
            logging.debug(f'Writing rate/observable information to {str(json_path)}')
            with json_path.open('w+') as f:
                json.dump({'observables': self.observables}, f)
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

    def get_input(self, rx_input: OpInput, idx: int | slice | np.ndarray):
        return rx_input.get(self._cache, idx)

    def get_inputs(self, rx_inputs: Sequence[OpInput], idx: int | slice | np.ndarray):
        return [self.get_input(x, idx) for x in rx_inputs]

    def get_input_idxs(self, rx_inputs: Sequence[OpInput]):
        return [self.rx_inputs.index(x) for x in rx_inputs]

    def make_split(self, name: str, new_idxs: np.ndarray) -> 'OpDataset':
        logging.info(f'Creating {name} split with {len(new_idxs)} samples')
        new_idxs = np.sort(new_idxs)
        json_path = self.json_path.with_name(f'{name}_{self.json_path.name}')
        json_path.unlink(missing_ok=True)
        with json_path.open('w+') as f:
            json.dump({'observables': self.observables}, f)
        new_ds = OpDataset(df=self.df.iloc[new_idxs],
                           observables=self.observables,
                           json_path=json_path)
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
    def concatenate(datasets: Sequence['OpDataset'], json_path: Optional[Path] = None) \
            -> 'OpDataset':
        if len(datasets) == 1:
            return datasets[0]
        logging.info(f'Concatenating {len(datasets)} OpDatasets...')
        ds0 = datasets[0]
        new_df = pd.concat([ds.df for ds in datasets]).reset_index(drop=True)
        json_path = json_path if json_path is not None else Path('/tmp/concat.json')
        json_path.unlink(missing_ok=True)
        with json_path.open('w+') as f:
            json.dump({'observables': ds0.observables}, f)
        new_ds = OpDataset(df=new_df,
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
                 rx_inputs: Optional[Sequence[OpInput]] = None,
                 preprocess_fn: Optional[Callable[[Path, pd.DataFrame], pd.DataFrame]] = None):
        datasets = []
        for p in path.glob('*.csv'):
            df = pd.read_csv(p)
            json_path = p.resolve().with_suffix('.json')
            json_path.unlink(missing_ok=True)
            dataset = OpDataset(df=df,
                                observables=['unedited', 'edited'],
                                json_path=json_path,
                                csv_path=p,
                                preprocess_fn=preprocess_fn,
                                rx_inputs=rx_inputs)
            datasets.append(dataset)
        return OpDataset.concatenate(datasets, json_path=(path / 'concat.json'))
