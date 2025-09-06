dN = list('ACGT')
rN = list('ACGU')
dNN = [f'{n1}{n2}' for n1 in dN for n2 in dN]
dNNN = [f'{n1}{n2}{n3}' for n1 in dN for n2 in dN for n3 in dN]
dNNNN = [f'{n1}{n2}{n3}{n4}' for n1 in dN for n2 in dN for n3 in dN for n4 in dN]

# RNA-DNA hybrid nearest-neighbor parameters from Banerjee, Sugimoto et al. (NAR 2020)
# Gibbs free energy (kcal/mol) in 100 mM NaCl at 37 C
RNA_DNA_dG = {'init': {'A': 2.6, 'C': 2.0, 'G': 2.0, 'U': 2.6},
              'stack': {'AA': -0.7, 'AC': -1.5, 'AG': -1.3, 'AU': -0.4,
                        'CA': -1.2, 'CC': -1.7, 'CG': -1.4, 'CU': -0.4,
                        'GA': -1.5, 'GC': -2.0, 'GG': -2.3, 'GU': -1.4,
                        'UA': -0.5, 'UC': -1.4, 'UG': -1.6, 'UU':  0.2} }
# RNA-RNA duplex nearest-neighbor parameters from Matthews et al. (PNAS 2004)
# Gibbs free energy (kcal/mol) in 100 mM NaCl at 37 C
RNA_RNA_dG = {'stack': {'AA': -0.9, 'AC': -2.2, 'AG': -2.1, 'AU': -1.1,
                        'CA': -2.1, 'CC': -3.3, 'CG': -2.4, 'CU': -2.1,
                        'GA': -2.4, 'GC': -3.4, 'GG': -3.3, 'GU': -2.2,
                        'UA': -1.3, 'UC': -2.4, 'UG': -2.1, 'UU': -0.9} }
# DNA-DNA duplex nearest-neighbor parameters from SantaLucia Jr et al. (Biochemistry 1996)
# Gibbs free energy (kcal/mol) in 100 mM NaCl at 37 C
DNA_DNA_dG = {'stack': {'AA': -1.02, 'AC': -1.43, 'AG': -1.16, 'AT': -0.73,
                        'CA': -1.38, 'CC': -1.77, 'CG': -2.09, 'CT': -1.16,
                        'GA': -1.46, 'GC': -2.28, 'GG': -1.77, 'GT': -1.43,
                        'TA': -0.60, 'TC': -1.46, 'TG': -1.38, 'TT': -1.02} }

HASH_SIZE = 10

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
