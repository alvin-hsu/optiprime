from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from Bio.SeqUtils import MeltingTemp as Mt
import RNA
from rs3.seq import predict_seq as rs3_predict

from .input_abc import OpInput, HashedScalarOpInput, HashedLargeOpInput

from optiprime.constants import CELL_TYPES, CAS9_PAMS, rN, dNN
from .utils import (np_stable_log, rna_fold, rna_fc, rna_distance, rna_to_dna,
                    delta_qs_scaffold, seq_encoding, duplex_bpp, total_pegrna_conc,
                    poison_pegrna_conc, EDIT_ENC_SIZE, PS20_OFFSET, POST_HOM_END)


####################
###  EDIT MODEL  ###
####################
@OpInput.decorator
class UneditedEncoding(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        max_u_len = df['full_unedited'].str.len().max()
        encodings = np.zeros((len(df), max_u_len, EDIT_ENC_SIZE), dtype=np.float32)
        for i, row in df.iterrows():
            enc, enc_len = seq_encoding(row, unedited=True)
            encodings[i, 0:enc_len, :] = enc
        cache[self.name] = encodings

    def concat(self, full_cache: Dict[str, Any], sub_caches: List[Dict[str, Any]]):
        concat_n = sum([sub_cache[self.name].shape[0] for sub_cache in sub_caches])
        max_len = max([sub_cache[self.name].shape[1] for sub_cache in sub_caches])
        enc_dim = sub_caches[0][self.name].shape[2]
        concat_enc = np.zeros((concat_n, max_len, enc_dim), dtype=np.float32)
        i = 0
        for sub_cache in sub_caches:
            entry = sub_cache[self.name]
            sub_n = entry.shape[0]
            sub_len = entry.shape[1]
            concat_enc[i:i + sub_n, 0:sub_len, :] = entry
            i += sub_n
        full_cache[self.name] = concat_enc

    def set_max_len(self, cache: Dict[str, Any], max_len: int):
        old_encodings = cache[self.name]
        ds_len, old_max, enc_size = old_encodings.shape
        assert old_max <= max_len, 'Can only make shape larger'
        new_encodings = np.zeros((ds_len, max_len, enc_size), dtype=old_encodings.dtype)
        new_encodings[:, :old_max, :] = old_encodings
        cache[self.name] = new_encodings

@OpInput.decorator
class EditedEncoding(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        max_e_len = df['full_edited'].str.len().max()
        encodings = np.zeros((len(df), max_e_len, EDIT_ENC_SIZE), dtype=np.float32)
        for i, row in df.iterrows():
            enc, enc_len = seq_encoding(row, unedited=False)
            encodings[i, 0:enc_len, :] = enc
        cache[self.name] = encodings

    def concat(self, full_cache: Dict[str, Any], sub_caches: List[Dict[str, Any]]):
        concat_n = sum([sub_cache[self.name].shape[0] for sub_cache in sub_caches])
        max_len = max([sub_cache[self.name].shape[1] for sub_cache in sub_caches])
        enc_dim = sub_caches[0][self.name].shape[2]
        concat_enc = np.zeros((concat_n, max_len, enc_dim), dtype=np.float32)
        i = 0
        for sub_cache in sub_caches:
            entry = sub_cache[self.name]
            sub_n = entry.shape[0]
            sub_len = entry.shape[1]
            concat_enc[i:i + sub_n, 0:sub_len, :] = entry
            i += sub_n
        full_cache[self.name] = concat_enc

    def set_max_len(self, cache: Dict[str, Any], max_len: int):
        old_encodings = cache[self.name]
        ds_len, old_max, enc_size = old_encodings.shape
        assert old_max <= max_len, 'Can only make shape larger'
        new_encodings = np.zeros((ds_len, max_len, enc_size), dtype=old_encodings.dtype)
        new_encodings[:, :old_max, :] = old_encodings
        cache[self.name] = new_encodings

@OpInput.decorator
class HetBPP(HashedLargeOpInput):
    dtype = np.float32
    mem_cached = False
    hash_column = 'pegrna_hash'

    def process_row(self, row: pd.Series) -> np.ndarray:
        return duplex_bpp(row)

    def pre_process(self):
        RNA.params_load_DNA_Mathews2004()

    def post_process(self):
        RNA.params_load_RNA_Turner2004()

    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        super().process(df, cache)
        max_u_len = df['full_unedited'].str.len().max()
        max_e_len = df['full_edited'].str.len().max()
        cache[self.meta_name]['max_u_len'] = max_u_len
        cache[self.meta_name]['max_e_len'] = max_e_len
        cache[self.meta_name]['shape'] = (max_u_len, max_e_len)

    def concat(self, full_cache: Dict[str, Any], sub_caches: List[Dict[str, Any]]):
        super().concat(full_cache, sub_caches)
        max_u_len = max([sub_cache[self.meta_name]['max_u_len'] for sub_cache in sub_caches])
        max_e_len = max([sub_cache[self.meta_name]['max_e_len'] for sub_cache in sub_caches])
        full_cache[self.meta_name]['max_u_len'] = max_u_len
        full_cache[self.meta_name]['max_e_len'] = max_e_len
        full_cache[self.meta_name]['shape'] = (max_u_len, max_e_len)

    def set_max_lens(self, cache: Dict[str, Any], max_u_len: int, max_e_len: int):
        assert cache[self.meta_name]['max_u_len'] <= max_u_len, 'Can only make shape larger'
        assert cache[self.meta_name]['max_e_len'] <= max_e_len, 'Can only make shape larger'
        cache[self.meta_name]['max_u_len'] = max_u_len
        cache[self.meta_name]['max_e_len'] = max_e_len
        cache[self.meta_name]['shape'] = (max_u_len, max_e_len)

@OpInput.decorator
class UneditedLen(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = df['full_unedited'].str.len().values

@OpInput.decorator
class EditedLen(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = df['full_edited'].str.len().values

@OpInput.decorator
class HomEndOneHot(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = np.zeros((len(df), 16), dtype=np.float32)
        idxs = df['full_edited'].apply(
            lambda x: dNN.index(x[-POST_HOM_END - 1:-POST_HOM_END + 1]))
        values[np.arange(len(df)), idxs.values] = 1
        cache[self.name] = values

@OpInput.decorator
class HomLen(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = df['post_hom'].str.len().values


EDIT_INPUTS = [UneditedEncoding, EditedEncoding, HetBPP, UneditedLen, EditedLen, HomEndOneHot, HomLen]
#####################
###  PE_ON MODEL  ###
#####################
@OpInput.decorator
class TotalPegConc(HashedScalarOpInput):
    hash_column = 'pegrna_hash'

    def process_row(self, row: pd.Series):
        return total_pegrna_conc(row)

@OpInput.decorator
class PoisonPegConc(HashedScalarOpInput):
    hash_column = 'pegrna_hash'

    def process_row(self, row: pd.Series):
        return poison_pegrna_conc(row)

@OpInput.decorator
class FirstRTTBase(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = np.zeros((len(df), 4), dtype=np.float32)
        idxs = df['rtt'].str.slice(0, 1).apply(lambda x: rN.index(x))
        values[np.arange(len(df)), idxs.values] = 1
        cache[self.name] = values

@OpInput.decorator
class UneditedPAMDA(OpInput):
    def __init__(self):
        fpath = Path(__file__).resolve().parent
        rate_df = pd.read_csv(fpath / 'HT-PAMDA.csv', index_col=0)
        self.pamda_data = rate_df.to_dict('dict')

    def get_pamda(self, row: pd.Series):
        cas, pam = row['cas9_pam'], row['u_pam']
        return self.pamda_data[cas][pam]

    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = df.apply(self.get_pamda, axis=1).values
        cache[self.name] = values

@OpInput.decorator
class PAMIndex(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = df['cas9_pam'].apply(CAS9_PAMS.index).values.astype(np.int32)

@OpInput.decorator
class ScaffoldDefect(HashedLargeOpInput):
    hash_column = 'pegrna_hash'
    dtype = np.float32
    mem_cached = True

    def process_row(self, row: pd.Series):
        dq_rar, dq_sl1, dq_sl2, dq_sl3 = delta_qs_scaffold(row)
        return np.array([dq_rar, dq_sl1, dq_sl2, dq_sl3], dtype=self.dtype)

    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        super().process(df, cache)
        cache[self.meta_name]['shape'] = (4,)

@OpInput.decorator
class UsingPEmax(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = (df['cas9_type'] != 'PE2-Cas9').values.astype(np.float32)

@OpInput.decorator
class UsingEPeg(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = (df['motif'] != 'none').values.astype(np.float32)

@OpInput.decorator
class UsingG21(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = (df['spacer'].str.len() == 21).values.astype(np.float32)

@OpInput.decorator
class Plus1GMisMatch(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = (df['spacer'].str.slice(0, 1) == 'g').values.astype(np.float32)

@OpInput.decorator
class RuleSet3Score(HashedScalarOpInput):
    hash_column = 'spacer_hash'

    def process_row(self, row: pd.Series):
        target = list(row['proto30'])
        target[25] = target[26] = 'G'  # Model doesn't account for PAM variants, so set to NGG
        target = ''.join(target)
        return rs3_predict([target], sequence_tracr='Chen2013')  # We account for poly(U) already

@OpInput.decorator
class PBSMeltRNA(HashedScalarOpInput):
    hash_column = 'pegrna_hash'

    def process_row(self, row: pd.Series):
        return Mt.Tm_NN(rna_to_dna(row['pbs']), nn_table=Mt.RNA_NN3)

@OpInput.decorator
class PBSMeltHybrid(HashedScalarOpInput):
    hash_column = 'pegrna_hash'

    def process_row(self, row: pd.Series):
        return Mt.Tm_NN(rna_to_dna(row['pbs']), nn_table=Mt.R_DNA_NN1)


@OpInput.decorator
class NickDinuc(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = np.zeros((len(df), 16), dtype=np.float32)
        idxs = df['full_unedited'].apply(lambda x: dNN.index(x[PS20_OFFSET + 16:PS20_OFFSET + 18]))
        values[np.arange(len(df)), idxs.values] = 1
        cache[self.name] = values


@OpInput.decorator
class PBSLen(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = np.zeros((len(df), 17), dtype=np.float32)
        idxs = df['pbs'].str.len() - 1
        values[np.arange(len(df)), idxs.values] = 1
        cache[self.name] = values


@OpInput.decorator
class PBSGC(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = np.zeros((len(df), 18), dtype=np.float32)
        idxs = df['pbs'].str.count('G') + df['pbs'].str.count('C')
        values[np.arange(len(df)), idxs.values] = 1
        cache[self.name] = values


PE_ON_INPUTS = [UneditedPAMDA, PAMIndex, PBSMeltRNA, PBSMeltHybrid, NickDinuc, PBSLen, PBSGC,
                TotalPegConc, FirstRTTBase, PoisonPegConc, ScaffoldDefect,
                UsingPEmax, UsingEPeg, UsingG21, Plus1GMisMatch, RuleSet3Score]


######################
###  PBS_ON_MODEL  ###
######################
@OpInput.decorator
class RTTDistance(HashedScalarOpInput):
    hash_column = 'pegrna_hash'
    dtype = np.float32
    mem_cached = True

    def process_row(self, row: pd.Series):
        rtt, pbs, linker = row['rtt'], row['pbs'], row['linker']
        return rna_distance(rtt + pbs + linker, 0, len(rtt))

@OpInput.decorator
class PBSFoldDefect(HashedScalarOpInput):
    hash_column = 'pegrna_hash'
    dtype = np.float32
    mem_cached = True

    def process_row(self, row: pd.Series) -> np.ndarray:
        rtt, pbs, linker = row['rtt'], row['pbs'], row['linker']
        ext = rtt + pbs + linker
        _, q_free = rna_fc(ext)
        cons = len(rtt) * '.' + len(pbs) * 'x' + len(linker) * '.'
        _, q_cons = rna_fc(ext, cons)
        return q_cons - q_free

@OpInput.decorator
class MeanPBSFreeLogP(HashedScalarOpInput):
    hash_column = 'pegrna_hash'
    dtype = np.float32
    mem_cached = True

    def process_row(self, row: pd.Series):
        rtt, pbs, linker = row['rtt'], row['pbs'], row['linker']
        bpp = rna_fold(rtt + pbs + linker)
        p = 0.0
        for i in range(len(rtt), len(rtt) + len(pbs)):
            p = p + bpp[i, :].sum() + bpp[:, i].sum()
        p = p / len(pbs)
        return np_stable_log(p)


PBS_ON_INPUTS = [RTTDistance, PBSFoldDefect, NickDinuc]


###################
###  SYN MODEL  ###
###################
@OpInput.decorator
class CellType(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        oh_enc = np.zeros((len(df), len(CELL_TYPES)), dtype=np.float32)
        for i, x in enumerate(CELL_TYPES):
            oh_enc[:, i] = df['group'].str.contains(x)
        other_mask = 1 - oh_enc.sum(axis=-1, keepdims=True)
        oh_enc = oh_enc + other_mask / len(CELL_TYPES)
        cache[self.name] = oh_enc

@OpInput.decorator
class RTTLen(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        rtt_lens = df['rtt'].str.len().values
        cache[self.name] = rtt_lens

@OpInput.decorator
class RTTBaseCounts(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        a_count = df['rtt'].str.count('A').astype(np.float32).values
        c_count = df['rtt'].str.count('C').astype(np.float32).values
        g_count = df['rtt'].str.count('G').astype(np.float32).values
        u_count = df['rtt'].str.count('U').astype(np.float32).values
        cache[self.name] = np.vstack([a_count, c_count, g_count, u_count]).transpose()

@OpInput.decorator
class RTTFoldEnergy(HashedScalarOpInput):
    hash_column = 'pegrna_hash'
    dtype = np.float32
    mem_cached = True

    def process_row(self, row: pd.Series) -> np.ndarray:
        _, q = rna_fc(row['rtt'])
        return q


SYN_INPUTS = [CellType, RTTLen, RTTBaseCounts, RTTFoldEnergy]


###################
###  MMR MODEL  ###
###################
@OpInput.decorator
class PEType(OpInput):
    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        cache[self.name] = (df['pe_type'] == 'PE2').astype(np.float32).values

@OpInput.decorator
class EditedPAMDA(OpInput):
    def __init__(self):
        fpath = Path(__file__).resolve().parent
        rate_df = pd.read_csv(fpath / 'HT-PAMDA.csv', index_col=0)
        self.pamda_data = rate_df.to_dict('dict')

    def get_pamda(self, row: pd.Series):
        cas, pam = row['cas9_pam'], row['e_pam']
        return self.pamda_data[cas][pam]

    def process(self, df: pd.DataFrame, cache: Dict[str, Any], verbose: bool = False):
        values = df.apply(self.get_pamda, axis=1).values
        cache[self.name] = values


MMR_INPUTS = [PEType, EditedPAMDA, NickDinuc, HomEndOneHot, HomLen, RuleSet3Score]
