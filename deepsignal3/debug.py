import numpy as np
import pod5 as p5


def extract_signal_from_pod5(pod5_path):
    # signals = {}
    num = 0
    flag = 0
    with p5.Reader(pod5_path) as reader:
        for read_record in reader.reads():
            # signals[str(read_record.read_id)] = {
            #    "signal": read_record.signal,
            #    "shift": read_record.calibration.offset,
            #    "scale": read_record.calibration.scale,
            # }  # 不加str会变成UUID，很奇怪
            num += 1
            flag += 1
            if flag == 1000000:
                flag = 0
                print('现在统计了{}条reads'.format(num))
    print('pod5 has {} reads'.format(num))
    return num  # output.pod5 1581750#说明该文件转化不完全
# 20221109_1654_5A_PAG65784_f306681d all.pod5 has 5887455 reads
# 20221109_1654_5D_PAG68757_39c39833 all.pod5 has 6568755 reads
# merge 12456210 reads from 2 files
#pod5_path = "/homeb/xiaoyf/data/HG002/20221109_1654_5D_PAG68757_39c39833/pod5/all.pod5"
#num = extract_signal_from_pod5(pod5_path)
def statistics(filename):
    total_data = -1
    with open(filename, "r") as f:
        total_data = len(f.readlines())
    print(str(filename)+' has {} line'.format(total_data))#30000000

#statistics('/home/xiaoyifu/data/HG002/R9.4/samples_CG.hc_poses.r30m.tsv')
import sys
import os
print(os.getcwd())

sys.path.append('/home/xiaoyifu/methylation/deepsignal/')
import numpy as np
from deepsignal3.utils import constants
def parse_a_line2(line):
    words = line.strip().split("\t")

    #sampleinfo = "\t".join(words[0:6])

    kmer = np.array([constants.base2code_dna[x] for x in words[6]])
    #base_means = np.array([float(x) for x in words[7].split(",")])
    #base_stds = np.array([float(x) for x in words[8].split(",")])
    #base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[10].split(";")])
    label = int(words[11])
    return kmer, k_signals, label

def prepare_data(filename):
    pos=0
    neg=0
    with open(filename, "r") as fr:
        line = fr.readline()
        while line:
            kmer, k_signals, label=parse_a_line2(line)
            if label==1:
                pos+=1
            elif label==0:
                neg+=1
            line = fr.readline()
    print('pos {} : neg {}'.format(pos,neg))#pos 896804 : neg 0
#prepare_data('/home/xiaoyifu/data/HG002/R9.4/fast5s.CG.features.tsv')
prepare_data('/home/xiaoyifu/data/HG002/R9.4/samples_CG.hc_poses.r30m.tsv')