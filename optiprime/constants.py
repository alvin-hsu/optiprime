PS20_OFFSET = 4
POST_HOM_END = 4

SCAFFOLD_NAMES = ['SpCas9_OG', 'OG_F+E', 'BlpI_F+E', 'GC_F+E']
SCAFFOLDS = {
    'SpCas9_OG': 'GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGC',
    'OG_F+E':    'GUUUAAGAGCUAUGCUGGAAACAGCAUAGCAAGUUUAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGC',
    'BlpI_F+E':  'GUUUAAGAGCUAAGCUGGAAACAGCAUAGCAAGUUUAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGC',
    'GC_F+E':    'GUUUCAGAGCUAUGCUGGAAACAGCAUAGCAAGUUGAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGC'}
RAR_CONSTRAINT = {
    'SpCas9_OG': '((((((..((((....))))....))))))..............................................',
    'OG_F+E':    '((((((..(((((((((....)))))))))....))))))..............................................',
    'BlpI_F+E':  '((((((..((((.((((....)))).))))....))))))..............................................',
    'GC_F+E':    '((((((..(((((((((....)))))))))....))))))..............................................'}
SL1_CONSTRAINT = {
    'SpCas9_OG': '................................((.....))...................................',
    'OG_F+E':    '..........................................((.....))...................................',
    'BlpI_F+E':  '..........................................((.....))...................................',
    'GC_F+E':    '..........................................((.....))...................................'}
SL2_CONSTRAINT = {
    'SpCas9_OG': '................................................((((....))))................',
    'OG_F+E':    '..........................................................((((....))))................',
    'BlpI_F+E':  '..........................................................((((....))))................',
    'GC_F+E':    '..........................................................((((....))))................'}
SL3_CONSTRAINT = {
    'SpCas9_OG': '............................................................x((((((...))))))',
    'OG_F+E':    '......................................................................x((((((...))))))',
    'BlpI_F+E':  '......................................................................x((((((...))))))',
    'GC_F+E':    '......................................................................x((((((...))))))'}

CAS9_NAMES = ['PE2-Cas9', 'PEmax-Cas9', 'PE6e-Cas9', 'PE6f-Cas9', 'PE6g-Cas9']
CAS9_PAMS = ['SpNGG', 'SpNG', 'SpNRCH', 'SpNRTH', 'SpNRRH', 'SpG', 'SpRY', 'SpVRQR', 'SpVQR', 'SpVRER']
PE_SYSTEMS = ['PE2', 'PE4']
RT_NAMES = ['PE2-RT', 'PE6a-RT', 'PE6b-RT', 'PE6c-RT', 'PE6d-RT']
MOTIF_NAMES = ['none', 'tevoPreQ1']
CELL_TYPES = ['HEK293T', 'HeLa', 'A549', 'HAP1', 'K562', 'U2OS', 'DLD1', 'MDA-MB-231', 'NIH3T3']
