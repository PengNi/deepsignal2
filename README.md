# DeepSignal3


## A deep-learning method for detecting methylation state from Oxford Nanopore reads.

#### For the VBZ compression issue
Please try adding ont-vbz-hdf-plugin to your environment as follows when all fast5s failed in `tombo resquiggle` and/or `deepsignal3 call_mods`. Normally it will work after setting `HDF5_PLUGIN_PATH`:
```shell
# download ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz (or newer version) and set HDF5_PLUGIN_PATH
# https://github.com/nanoporetech/vbz_compression/releases
wget https://github.com/nanoporetech/vbz_compression/releases/download/v1.0.1/ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz
tar zxvf ont-vbz-hdf-plugin-1.0.1-Linux-x86_64.tar.gz
export HDF5_PLUGIN_PATH=/abslolute/path/to/ont-vbz-hdf-plugin-1.0.1-Linux/usr/local/hdf5/lib/plugin
```


## Contents
- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Quick start](#Quick-start)
- [Usage](#Usage)

## Installation
deepsignal3 is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/). 
   - Prerequisites:\
       [Python3.*](https://www.python.org/) \
       [Guppy](https://nanoporetech.com/community)
   - Dependencies: \
       [numpy](http://www.numpy.org/) \
       [h5py](https://github.com/h5py/h5py) \
       [statsmodels](https://github.com/statsmodels/statsmodels/) \
       [scikit-learn](https://scikit-learn.org/stable/) \
       [mappy](https://github.com/lh3/minimap2/tree/master/python) \
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.11.0)

#### 1. Create an environment
We highly recommend to use a virtual environment for the installation of deepsignal3 and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):
```bash
# create
conda create -n deepsignalpenv python=3.8
# activate
conda activate deepsignalpenv
# deactivate
conda deactivate
```
The virtual environment can also be created by using [virtualenv](https://github.com/pypa/virtualenv/).

#### 2. Install deepsignal3
- After creating and activating the environment, download deepsignal3 (**lastest version**) from github:
```bash
git clone https://github.com/PengNi/deepsignal2.git
cd deepsignal2
python setup.py install
```

- [PyTorch](https://pytorch.org/) can be automatically installed during the installation of deepsignal3. However, if the version of [PyTorch](https://pytorch.org/) installed is not appropriate for your OS, an appropriate version should be re-installed in the same environment as the [instructions](https://pytorch.org/get-started/locally/):
```bash
# install using conda
conda install pytorch==1.11.0 cudatoolkit=10.2 -c pytorch
# or install using pip
pip install torch==1.11.0
```


## Trained models
Currently, we have trained the following models:
   * [hg002.rmet0.95_0.05.r10.4.10x.CG.epoch8.ckpt](https://github.com/PengNi/deepsignal2/blob/r10.4/model/model.r10.CG/hg002.rmet0.95_0.05.r10.4.10x.CG.epoch8.ckpt): model trained using HG002 R10.4 with chm13v2 as reference genome.
## Quick start
To call modifications, the raw fast5 files should be basecalled ([Guppy](https://nanoporetech.com/community)(version <=6.2.1)). Belows are commands to call 5mC in CG, CHG, and CHH contexts:
```bash
# Higher versions of Guppy no longer support the output format fast5
# Download and unzip the example data and pre-trained models.
# 1. guppy basecall using GPU
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r10.4.1_e8.2_400bps_hac_prom.cfg --device CUDA:0 --fast5_out
# 2. deepsignal3 call_mods
# CG
CUDA_VISIBLE_DEVICES=0 deepsignal3 call_mods --input_path fast5s_guppy --model_path *.ckpt --result_file fast5s.CG.call_mods.tsv --reference_path chm13v2.0.fa --motifs CG --nproc 30 --nproc_gpu 6
deepsignal3 call_freq --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.tsv
```


## Usage
#### 1. Basecall and re-squiggle
Before run deepsignal, the raw reads should be basecalled ([Guppy](https://nanoporetech.com/community)(version <=6.2.1)).

For the example data:
```bash
# 1. basecall using GPU
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r10.4.1_e8.2_400bps_hac_prom.cfg --device CUDA:0 --fast5_out
# or using CPU
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r10.4.1_e8.2_400bps_hac_prom.cfg --fast5_out
```

#### 2. extract features
Features of targeted sites can be extracted for training or testing.

For the example data (deepsignal3 extracts 13-mer-seq and 13*15-signal features of each CpG motif in reads by default.:
```bash
deepsignal3 extract -i fast5s_guppy --reference_path chm13v2.0.fa -o fast5s.CG.features.tsv --nproc 30 --motifs CG &
```

The extracted_features file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **readname**:  the read name
   - **read_strand**:   t/c, template or complement
   - **k_mer**: the sequence around the targeted base
   - **signal_means**:  signal means of each base in the kmer
   - **signal_stds**:   signal stds of each base in the kmer
   - **signal_lens**:   lens of each base in the kmer
   - **raw_signals**:  signal values for each base of the kmer, splited by ';'
   - **methy_label**:   0/1, the label of the targeted base, for training

#### 3. call modifications

To call modifications, either the extracted-feature file or **the raw fast5 files (recommended)** can be used as input.

For the example data:
```bash
# call 5mCpGs for instance

# extracted-feature file as input
deepsignal3 call_mods --input_path fast5s.CG.features.tsv --model_path hg002.rmet0.95_0.05.r10.4.10x.CG.epoch8.ckpt --result_file fast5s.CG.call_mods.tsv --motifs CG --nproc 30 --nproc_gpu 6

# fast5 files as input, use GPU
CUDA_VISIBLE_DEVICES=0 deepsignal3 call_mods --input_path fast5s_guppy --model_path hg002.rmet0.95_0.05.r10.4.10x.CG.epoch8.ckpt --result_file fast5s.CG.call_mods.tsv --reference_path chm13v2.0.fa --motifs CG --nproc 30 --nproc_gpu 6
```

The modification_call file is a tab-delimited text file in the following format:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **readname**:  the read name
   - **read_strand**:   t/c, template or complement
   - **prob_0**:    [0, 1], the probability of the targeted base predicted as 0 (unmethylated)
   - **prob_1**:    [0, 1], the probability of the targeted base predicted as 1 (methylated)
   - **called_label**:  0/1, unmethylated/methylated
   - **k_mer**:   the kmer around the targeted base

#### 4. call frequency of modifications
A modification-frequency file can be generated by `call_freq` function with the call_mods file as input:
```bash
# call 5mCpGs for instance

# output in tsv format
deepsignal3 call_freq --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.tsv
# output in bedMethyl format
deepsignal3 call_freq --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.bed --bed
# use --sort to sort the results
deepsignal3 call_freq --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.bed --bed --sort
```

The modification_frequency file can be either saved in [bedMethyl](https://www.encodeproject.org/data-standards/wgbs/) format (by setting `--bed` as above), or saved as a tab-delimited text file in the following format by default:
   - **chrom**: the chromosome name
   - **pos**:   0-based position of the targeted base in the chromosome
   - **strand**:    +/-, the aligned strand of the read to the reference
   - **pos_in_strand**: 0-based position of the targeted base in the aligned strand of the chromosome
   - **prob_0_sum**:    sum of the probabilities of the targeted base predicted as 0 (unmethylated)
   - **prob_1_sum**:    sum of the probabilities of the targeted base predicted as 1 (methylated)
   - **count_modified**:    number of reads in which the targeted base counted as modified
   - **count_unmodified**:  number of reads in which the targeted base counted as unmodified
   - **coverage**:  number of reads aligned to the targeted base
   - **modification_frequency**:    modification frequency
   - **k_mer**:   the kmer around the targeted base

#### 5. train new models
A new model can be trained as follows:
```bash
# need to split training samples to two independent datasets for training and validating
# please use deepsignal3 train -h/--help for more details
deepsignal3 train --train_file /path/to/train/file --valid_file /path/to/valid/file --model_dir /dir/to/save/the/new/model
```

License
=========
Copyright (C) 2020 [Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Feng Luo](mailto:luofeng@clemson.edu), [Peng Ni](mailto:nipeng@csu.edu.cn),[Yifu Xiao](mailto:234701005@csu.edu.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

[Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Peng Ni](mailto:nipeng@csu.edu.cn),[Yifu Xiao](mailto:234701005@csu.edu.cn)
School of Computer Science and Engineering, Central South University, Changsha 410083, China

[Feng Luo](mailto:luofeng@clemson.edu), School of Computing, Clemson University, Clemson, SC 29634, USA
