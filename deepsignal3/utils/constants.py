#When modifying a constant (variable) in one file (A.py) and another file (B.py), if you import B.py and access the constant in C.py, the value returned is the value before the modification, because the Python module caches its content after the first import and does not reload. This means that if you modify the constants in B.py in A.py, but C.py has already imported B.py, C.py will still use the values from the initial import of B.py unless you re import B.py
iupac_alphabets = {
    "A": ["A"],
    "T": ["T"],
    "C": ["C"],
    "G": ["G"],
    "R": ["A", "G"],
    "M": ["A", "C"],
    "S": ["C", "G"],
    "Y": ["C", "T"],
    "K": ["G", "T"],
    "W": ["A", "T"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}
iupac_alphabets_rna = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "U": ["U"],
    "R": ["A", "G"],
    "M": ["A", "C"],
    "S": ["C", "G"],
    "Y": ["C", "U"],
    "K": ["G", "U"],
    "W": ["A", "U"],
    "B": ["C", "G", "U"],
    "D": ["A", "G", "U"],
    "H": ["A", "C", "U"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "U"],
}


base2code_dna = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
code2base_dna = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

POD5_READ_ID = 0
POD5_SIGNAL = 1
# POD5_SHIFT = 2
# POD5_SCALE = 3

BAM_SEQ = 1
BAM_STRIDE = 2
BAM_MV_TABLE = 3
BAM_NUM_TRIMMED = 4
BAM_TO_NORM_SHIFT = 5
BAM_TO_NORM_SCALE = 6
BAM_REF_NAME = 7
BAM_REF_START = 8
BAM_REF_STRAND = 9

READ_ID = 0
READ_SIGNAL = 1
# READ_TO_PA_SHIFT = 2
# READ_TO_PA_SCALE = 3
READ_SEQ = 2
READ_STRIDE = 3
READ_MV_TABLE = 4
READ_NUM_TRIMMED = 5
READ_TO_NORM_SHIFT = 6
READ_TO_NORM_SCALE = 7
READ_REF_NAME = 8
READ_REF_START = 9
READ_REF_STRAND = 10

FEATURE_ID = 0
FEATURE_SIG = 1
FEATURE_SEQ = 2
FEATURE_BASE_TO_SIG_NUM = 3
FEATURE_REF_NAME = 4
FEATURE_REF_START = 5
FEATURE_REF_STRAND = 6

SIG_LEN=10
KMER_LEN=17