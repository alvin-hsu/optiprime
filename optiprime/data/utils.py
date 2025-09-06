import logging
from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import RNA

from optiprime.constants import *

##############
###  MISC  ###
##############
EPS = 1e-5
def np_stable_log(x, eps=EPS):
    return np.log((x + eps) / (1 + eps))


_RC_TRANS = str.maketrans('ACGT', 'tgca')
def revcomp_dna(dna: str):
    return dna.translate(_RC_TRANS)[::-1].upper()


def dna_to_rna(dna: str) -> str:
    return dna.replace('T', 'U')


def rna_to_dna(rna: str) -> str:
    return rna.replace('U', 'T')


def rna_fold(seq: str) -> np.ndarray:
    """
    Returns a dense array with the base-pairing probability between two bases in SEQ.
    """
    fc, _ = rna_fc(seq=seq)
    return np.array(fc.bpp(), dtype=np.float32)[1:, 1:]


def rna_distance(seq: str, start: int, end: int) -> float:
    def distance(db: str):
        dists = np.abs(np.arange(len(seq)) - start)
        d_bp_stack = []
        prev_d = 10000
        # Rightward update
        for i, (x, d) in enumerate(zip(db, dists)):
            # Update based on adjacency
            d = min(d, prev_d + 1)
            # Manage stack for base pairing
            if x == '(':
                d_bp_stack.append(d)
            elif x == ')':
                new_d = d_bp_stack.pop() + 1
                d = min(d, new_d)  # If base pairing gives better distance
            dists[i] = prev_d = d
        # Leftward update
        prev_d = 10000
        for i in range(len(seq) - 1, -1, -1):
            d = dists[i]
            d = min(d, prev_d + 1)
            dists[i] = prev_d = d
        return dists[end]

    fc, _ = rna_fc(seq=seq)
    samples = fc.pbacktrack(100)
    distances = [distance(db) for db in samples]
    return np.mean(distances)


def rna_fc(seq: str, constraint: Optional[str] = None):
    # Enable sampling from Boltzmann ensemble for RNA folding
    md = RNA.md()
    md.uniq_ML = 1  # noqa
    fc = RNA.fold_compound(seq, md, RNA.OPTION_PF)
    if constraint is not None:
        fc.hc_add_from_db(constraint)
    _, z = fc.pf()
    return fc, z


################################
###  PROCESSING BATCH FILES  ###
################################
_REQUIRED_COLUMNS = ['spacer', 'rtt', 'pbs', 'full_unedited', 'full_edited']
_DEFAULT_VALUES = {'scaffold_name': 'SpCas9_OG',
                   'linker': '',
                   'motif': 'tevoPreQ1',
                   'cas9_type': 'PE2-Cas9',
                   'cas9_pam': 'SpNGG',
                   'rt_name': 'PE2-RT',
                   'pe_type': 'PE2',
                   'group': 'NO_GROUP',
                   'time': 1.0,
                   'split': np.nan,
                   'weight': 1.0,
                   'edited_frac': 0.0,
                   'indel_frac': 0.0}
_ALL_COLUMNS = ['spacer', 'scaffold_name', 'rtt', 'pbs', 'linker', 'motif',         # pegRNA
                'full_unedited', 'full_edited',                                     # edit
                'cas9_type', 'cas9_pam', 'rt_name', 'pe_type',                      # editor
                'group', 'time', 'split', 'weight', 'unedited', 'edited', 'indel']  # other
def format_pe_df(p: Path, df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)  # Renumbers everything
    # Check for required columns
    for col_name in _REQUIRED_COLUMNS:
        assert col_name in df.columns, f'{col_name} must be explicitly specificed ({p})'
    # Set missing columns to defaults
    for col_name, default in _DEFAULT_VALUES.items():
        if col_name not in df.columns:
            logging.info(f'{col_name} not found, defaulting to {default}')
            df[col_name] = default
    # Calculate unedited fraction
    df['unedited'] = 1 - (df['edited_frac'] + df['indel_frac'])
    df = df.rename(columns={'edited_frac': 'edited',
                            'indel_frac': 'indel'})
    # Remove weight-0 or invalid entries from dataset for efficiency
    df = df[df['weight'] > 0]
    df = df.dropna(subset=['unedited', 'edited', 'weight']).reset_index()
    df = df.fillna('')
    # Add columns that we can make automatically
    # Usually we'll have at least 30 bases, but have the option of using this
    # for those edits where the homology length is absurdly short
    if 'proto30' not in df.columns:
        df['proto30'] = df['full_unedited'].str.slice(0, 30)  # Check PS20_OFFSET?
    assert (df['proto30'].str.len() == 30).all(), f'Wrong proto30 len ({p})'
    df['spacer'] = df['spacer'].str.replace('T', 'U')
    df['rtt'] = df['rtt'].str.replace('T', 'U')
    df['pbs'] = df['pbs'].str.replace('T', 'U')
    df['pegrna'] = (df['spacer'] + df['scaffold_name'].apply(lambda x: SCAFFOLDS[x]) +
                    df['rtt'] + df['pbs'] + df['linker'])
    df[['pre_hom', 'min_edit', 'post_hom']] = df.apply(split_edit, axis=1, result_type='expand')
    df[['min_u', 'min_e']] = df['min_edit'].str.split(':', expand=True)
    df[['u_pam', 'e_pam']] = df.apply(u_e_pam, axis=1, result_type='expand')
    df['seed_edit'] = df.apply(seed_edit, axis=1)
    df['hom_len'] = df['post_hom'].str.len()
    return df


def split_edit(row: pd.Series) -> str:
    unedited = row['full_unedited']
    edited = row['full_edited']
    i = 0
    for i, (u, e) in enumerate(zip(unedited, edited)):
        if u != e:
            break
    pre_hom = unedited[:i]
    unedited = unedited[i:]
    edited = edited[i:]
    i = 0
    for i, (u, e) in enumerate(zip(unedited[::-1], edited[::-1])):
        if u != e:
            break
    else:
        i = i + 1
    if i:
        post_hom = unedited[-i:]
        unedited = unedited[:-i]
        edited = edited[:-i]
    else:
        post_hom = ''
    return pre_hom, unedited + ':' + edited, post_hom


def u_e_pam(row: pd.Series) -> Tuple[str, str]:
    u_pam = row['proto30'][PS20_OFFSET + 20:PS20_OFFSET + 24]
    edited = row['full_edited']
    if len(edited) > PS20_OFFSET + 20:
        e_pam = list(u_pam)
        for i in range(min(len(edited) - (PS20_OFFSET + 20), 4)):
            e_pam[i] = edited[PS20_OFFSET + 20 + i]
        return u_pam, ''.join(e_pam)
    else:
        return u_pam, u_pam


def seed_edit(row: pd.Series) -> str:
    old = row['full_unedited'][PS20_OFFSET + 17:PS20_OFFSET + 20]
    new = row['full_edited'][PS20_OFFSET + 17:PS20_OFFSET + 20]
    diff = 0
    for u, e in zip(old, new):
        if u != e:
            diff += 1
    return diff


# Sibghat-Ullah et al., Biochemistry 1996 Figure 6
POST_TG_SCORES = {'A': 0.33, 'C': 0.30, 'G': 1.00, 'T': 0.08}
def tdg_scores(row: pd.Series) -> Tuple[float, float, float, float]:
    if len(row['min_u']) == 0 or len(row['min_e']) == 0:
        return 0, 0, 0, 0
    elif row['min_edit'] == 'C:T':
        return POST_TG_SCORES[row['post_hom'][0]], 0, 0, 0
    elif row['min_edit'] == 'A:G':
        return 0, POST_TG_SCORES[revcomp_dna(row['pre_hom'][-1])], 0, 0
    else:
        start_c_t = end_a_g = 0
        if (row['min_u'][0] == 'C') and (row['min_e'][0] == 'T'):
            start_c_t = 1
        if (row['min_u'][-1] == 'A') and (row['min_e'][-1] == 'G'):
            end_a_g = 1
        return 0, 0, start_c_t, end_a_g


#########################################
###  CALCULATE PE-RELATED QUANTITIES  ###
#########################################
# Premature polyU termination => decreased total viable pegRNA conc.
_POLYU_FRAC = [None, 1., 1., 1., 0.25, 0.05, 0.01, 0.]  # Frac. of pegRNAs terminated after Us
_POLYU_RE = re.compile('U+')
def total_pegrna_conc(row: pd.Series):
    pegrna_seq = row['pegrna']
    frac_conc = 1.0
    if row.get('pol3', True):
        for match in _POLYU_RE.findall(pegrna_seq):
            u_len = min(len(match), 7)
            frac_conc *= _POLYU_FRAC[u_len]
    else:
        frac_conc = 0.98**len(pegrna_seq)
    return frac_conc

# Premature polyU termination *after scaffold* => poison pegRNA species
def poison_pegrna_conc(row: pd.Series):
    ext_seq = row['rtt'] + row['pbs']
    frac_conc = 1.0
    if row.get('pol3', True):
        for match in _POLYU_RE.findall(ext_seq):
            u_len = min(len(match), 7)
            frac_conc *= _POLYU_FRAC[u_len]
    else:
        frac_conc = 0.98**len(ext_seq)
    return 1.0 - frac_conc


def delta_qs_scaffold(row: pd.Series) -> Tuple[float, float, float, float]:
    spacer, rtt, pbs, linker = row['spacer'], row['rtt'], row['pbs'], row['linker']
    scaffold_name = row['scaffold_name']
    pegrna_seq = row['pegrna']
    _, q_free = rna_fc(pegrna_seq)
    spacer_cons = len(spacer) * '.'
    ext_cons = (len(rtt) + len(pbs) + len(linker)) * '.'
    _, q_rar = rna_fc(pegrna_seq, spacer_cons + RAR_CONSTRAINT[scaffold_name] + ext_cons)
    _, q_sl1 = rna_fc(pegrna_seq, spacer_cons + SL1_CONSTRAINT[scaffold_name] + ext_cons)
    _, q_sl2 = rna_fc(pegrna_seq, spacer_cons + SL2_CONSTRAINT[scaffold_name] + ext_cons)
    _, q_sl3 = rna_fc(pegrna_seq, spacer_cons + SL3_CONSTRAINT[scaffold_name] + ext_cons)
    return q_rar - q_free, q_sl1 - q_free, q_sl2 - q_free, q_sl3 - q_free


def sim_mh_anneal(row: pd.Series):
    full_u, full_e = row['full_unedited'], row['full_edited']
    start, u_end, e_end = PS20_OFFSET + 17, len(full_u) - POST_HOM_END, len(full_e) - POST_HOM_END
    unedited, flap3 = row['full_unedited'][start:u_end], row['full_edited'][start:e_end]
    # Strands for homology folding
    bottom_strand = revcomp_dna(full_u)
    flap3_strand = full_u[PS20_OFFSET:PS20_OFFSET + 17] + flap3
    # Offsets
    f3_len = len(flap3_strand)
    b_len = len(bottom_strand)
    b_off = 1
    f3_off = b_off + b_len
    u_off = f3_off + f3_len
    # Keeping track of indices
    up_i = 0  # Unpaired len
    true_f_i = 0  # "True" flap index
    n2_dists = {k: [] for k in dNN}
    while len(unedited) > 1 and len(flap3) > 1:
        # Find distance to dinucleotide "microhomology"
        mh = unedited[0:2]
        if mh == flap3[up_i:up_i + 2]:
            dist = 0
        else:
            for i in range(1, len(flap3)):
                if up_i + i + 2 <= len(flap3) and mh == flap3[up_i + i:up_i + i + 2]:
                    dist = i
                    break
                elif up_i - i >= 0 and mh == flap3[up_i - i:up_i - i + 2]:
                    dist = -i
                    break
            else:
                dist = None
        # If microhomology present, find maximum homology resulting from microhomology
        if dist is not None:
            n2_dists[mh].append(dist)
            i = 0
            for i, (u, f) in enumerate(zip(unedited, flap3[up_i + dist:])):
                if u != f:
                    break
            else:
                i = i + 1
            h_len = i
            unedited = unedited[h_len:]
            u_len = len(unedited)
            if u_len == 0:
                break
            # Check if homology is sufficient to stay paired
            fc = RNA.fold_compound(f'{bottom_strand}&{flap3_strand}&{unedited}')
            fc.hc_add_bp(b_off + 0, u_off + u_len - 1)
            fc.hc_add_bp(b_off + u_len + h_len + up_i, f3_off + 17 + true_f_i - 1)
            fc.hc_add_bp(b_off + b_len - 1, f3_off + 0)
            st, _ = fc.pf()
            bpp = np.array(fc.bpp())
            # Base pairing probability for each base of homology
            h_bpp = 0.0
            for i in range(h_len):
                h_bpp += bpp[b_off + u_len + i,
                             f3_off + 17 + true_f_i + up_i + dist + h_len - i - 1]
            h_bpp = h_bpp / h_len
            if h_bpp > 0.9:
                true_f_i = true_f_i + up_i + dist + h_len
                flap3 = flap3[up_i + dist + h_len:]
                up_i = 0
            else:
                up_i = up_i + h_len
        else:  # No homology found
            n2_dists[mh].append(None)
            unedited = unedited[1:]
            up_i += 1
    return n2_dists


###################################
###  ENCODINGS FOR DEEP MODELS  ###
###################################
EDIT_ENC_SIZE = 9
def seq_encoding(row: pd.Series, unedited: bool) -> Tuple[np.ndarray, int]:
    """
    Returns a simple encoding of the minimal edited/unedited sequence. Indices (inclusive) are
    given below:
        BASES (0-3):
          0: A   1: C   2: G   3: T
        SEQUENCE LOCATIONS (4-7):
          4: pre-homology   5: unedited   6: edited   7: post-homology
        MASK (8):
          8: <MASK> for pretraining.
    """
    seq = row['full_unedited'] if unedited else row['full_edited']
    encoding = np.zeros((len(seq), EDIT_ENC_SIZE), dtype=np.float32)
    for i, x in enumerate(seq):
        encoding[i, dN.index(x)] = 1
    pre_i, post_i = len(row['pre_hom']), len(row['post_hom'])
    encoding[:pre_i, 4] = 1
    seq_i = 5 if unedited else 6
    encoding[pre_i:post_i, seq_i] = 1
    encoding[post_i:, 7] = 1
    return encoding, len(seq)


def duplex_bpp(row: pd.Series) -> np.ndarray:
    unedited, edited = row['full_unedited'], row['full_edited']
    pre_hom_len, post_hom_len = len(row['pre_hom']), len(row['post_hom'])
    unedited_rc = revcomp_dna(unedited)
    e_cons = ['.' for _ in range(len(edited))]
    u_cons = ['.' for _ in range(len(unedited))]
    for i in range(pre_hom_len - 1):
        e_cons[i] = '('
        u_cons[i] = ')'
    for i in range(post_hom_len - 1):
        e_cons[-(1 + i)] = '('
        u_cons[-(1 + i)] = ')'
    e_cons = ''.join(e_cons)
    u_cons = ''.join(u_cons[::-1])
    fc = RNA.fold_compound(f'{edited}&{unedited_rc}')
    fc.hc_add_from_db(e_cons + u_cons)
    fc.pf()
    bpp = np.array(fc.bpp())[1:, 1:]
    bpp = bpp[:len(edited), len(edited):]
    return bpp[:, ::-1].transpose()
