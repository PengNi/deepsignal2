from torch.utils.data import Dataset
import linecache
import os
import numpy as np
from utils import constants


def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()


def parse_a_line(line):
    words = line.strip().split("\t")

    seq = np.array(
        [[constants.base2code_dna[y] for y in x.split(",")] for x in words[1].split(";")]
    )
    signal = np.array(
        [[np.float16(y) for y in x.split(",")] for x in words[2].split(";")]
    )
    label = np.random.randint(0, 2)
    # if rlabel==1:
    #    label=[0,1]
    # else:
    #    label=[1,0]

    return seq, signal, label


class SignalDataset(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


def parse_a_line2(line):
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([constants.base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[10].split(";")])
    label = int(words[11])
    return kmer, k_signals, label


def prepare_dataset(filename, target_chr) -> str:
    chr_filename = str(os.path.dirname(filename)) + "/" + target_chr + ".txt"
    with open(chr_filename, "w") as fw:
        with open(filename, "r") as fr:
            line = fr.readline()
            while line:
                words = line.strip().split("\t")
                chr = words[0]
                # print(chr)
                if chr == target_chr:
                    fw.write(line)
                line = fr.readline()
    return chr_filename


class SignalFeaData2(Dataset):
    def __init__(self, filename, target_chr="chr11", transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        if target_chr != 'all':
            if os.path.exists(str(os.path.dirname(filename)) + "/" + target_chr + ".txt"):
                filename = str(os.path.dirname(filename)) + "/" + target_chr + ".txt"
            else:
                filename = prepare_dataset(filename, target_chr)
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line2(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data
