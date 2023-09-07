from torch.utils.data import Dataset
import linecache
import os
import numpy as np
from utils import constants
import gc

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

    #sampleinfo = "\t".join(words[0:6])

    kmer = np.array([[constants.base2code_dna[x]]*constants.SIG_LEN for x in words[6]])
    #base_means = np.array([float(x) for x in words[7].split(",")])
    #base_stds = np.array([float(x) for x in words[8].split(",")])
    #base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    k_signals = np.array([[float(y) for y in x.split(",")][3:-3] for x in words[10].split(";")])
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

def count_line_num(sl_filepath, fheader=False):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    # print('done count the lines of file {}'.format(sl_filepath))
    return count

def generate_offsets(filename):
    offsets = []
    with open(filename, "r") as rf:
        offsets.append(rf.tell())
        while rf.readline():
            offsets.append(rf.tell())
    return offsets

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
        self._total_data = count_line_num(filename, False)
        self._offsets = generate_offsets(filename)
        self._current_offset = 0
        #with open(filename, "r") as f:
        #    self._total_data = len(f.readlines())#这里会读取整个文件，而且似乎在多GPU并行中超级加倍了，真坑

        gc.collect()

    def __getitem__(self, idx):
        #line = linecache.getline(self._filename, idx + 1)
        #if line == "":
        #    return None
        #else:
        #    output = parse_a_line2(line)
        #    if self._transform is not None:
        #        output = self._transform(output)
        #    return output
        offset = self._offsets[idx]
        # self._data_stream.seek(offset)
        # line = self._data_stream.readline()
        with open(self._filename, "r") as rf:
            rf.seek(offset)
            line = rf.readline()
        output = parse_a_line2(line)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data
