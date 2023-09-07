import linecache
import os
import argparse

def split_dataset(filename, number):
    target_filename = str(os.path.dirname(filename)) + "/" + str(number) + ".tsv"
    num=0
    with open(target_filename, "w") as fw:
        with open(filename, "r") as fr:
            line = fr.readline()
            while line:
                num+=1
                # print(chr)
                fw.write(line)
                if num>=number:
                    break    
                line = fr.readline()
def parse_args():
    parser = argparse.ArgumentParser('split dataset to test')
    parser.add_argument(
        "--input-file",'-i',
        type=str,
        default="/home/xiaoyifu/data/HG002/R9.4/fast5s_new.CG.features.tsv",
        help="raw dataset adress.",
    )
    parser.add_argument(
        "--dataset-size",'-d',
        type=int,
        default=1000000,
        help="size of dataset used to test.",
    )
    return parser.parse_args()
if __name__ == '__main__':
    args=parse_args()
    split_dataset(args.input_file,args.dataset_size)