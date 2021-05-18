# DeepSignal2

## A deep-learning method for detecting methylation state from Oxford Nanopore sequencing reads.
**deepsignal2 has the same DNN structure with [deepsignal-plant](https://github.com/PengNi/deepsignal-plant), so the pre-trained models of deepsignal2/deepsignal-plant can be used by both tools. Importantly, when using models of these two tools, note that default `--seq_len` in deepsignal2 is 17, while in deepsignal-plant is 13.**

deepsignal2 applys BiLSTM to detect methylation from Nanopore reads. It is built on **Python3** and **PyTorch**.

## Contents
- [Installation](#Installation)
- [Trained models](#Trained-models)
- [Quick start](#Quick-start)
- [Usage](#Usage)

## Installation
deepsignal2 is built on [Python3](https://www.python.org/) and [PyTorch](https://pytorch.org/). [tombo](https://github.com/nanoporetech/tombo) is required to re-squiggle the raw signals from nanopore reads before running deepsignal2.
   - Prerequisites:\
       [Python3.*](https://www.python.org/)\
       [tombo](https://github.com/nanoporetech/tombo) (version 1.5.1)
   - Dependencies:\
       [numpy](http://www.numpy.org/)\
       [h5py](https://github.com/h5py/h5py)\
       [statsmodels](https://github.com/statsmodels/statsmodels/)\
       [scikit-learn](https://scikit-learn.org/stable/)\
       [PyTorch](https://pytorch.org/) (version >=1.2.0, <=1.5.0?)

#### 1. Create an environment
We highly recommend to use a virtual environment for the installation of deepsignal2 and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):
```bash
# create
conda create -n deepsignal2env python=3.6
# activate
conda activate deepsignal2env
# deactivate
conda deactivate
```
The virtual environment can also be created by using [virtualenv](https://github.com/pypa/virtualenv/).

#### 2. Install deepsignal2
- After creating and activating the environment, download deepsignal2 (**lastest version**) from github:
```bash
git clone https://github.com/PengNi/deepsignal2.git
cd deepsignal2
python setup.py install
```
**or** install deepsignal2 using *pip*:
```bash
pip install deepsignal2
```

- [PyTorch](https://pytorch.org/) can be automatically installed during the installation of deepsignal2. However, if the version of [PyTorch](https://pytorch.org/) installed is not appropriate for your OS, an appropriate version should be re-installed in the same environment as the [instructions](https://pytorch.org/get-started/locally/):
```bash
# install using conda
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
# or install using pip
pip install torch==1.4.0 torchvision==0.5.0
```

- [tombo](https://github.com/nanoporetech/tombo) is required to be installed in the same environment:
```bash
# install using conda
conda install -c bioconda ont-tombo
# or install using pip
pip install ont-tombo
```


## Trained models
The models we trained can be downloaded from [google drive](https://drive.google.com/drive/folders/1F8tImt4aMC4HPznhjczUdMJ64mre8dG9?usp=sharing).

Currently we have trained the following models:
   * _model.dp2.CG.R9.4_1D.human_hx1.bn17_sn16.both_bilstm.b17_s16_epoch4.ckpt_: A CpG (5mC) model trained using human HX1 R9.4 1D reads.


## Quick start
To call modifications, the raw fast5 files should be basecalled ([Guppy>=3.6.1](https://nanoporetech.com/community)) and then be re-squiggled by [tombo](https://github.com/nanoporetech/tombo). At last, modifications of specified motifs can be called by deepsignal. Belows are commands to call 5mC in CG contexts:
```bash
# 1. guppy basecall
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r9.4.1_450bps_hac_prom.cfg
# 2. tombo resquiggle
cat fast5s_guppy/*.fastq > fast5s_guppy.fastq
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5s/ --fastq-filenames fast5s_guppy.fastq --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10
tombo resquiggle fast5s/ /path/to/genome/reference.fa --processes 10 --corrected-group RawGenomeCorrected_000 --basecall-group Basecall_1D_000 --overwrite
# 3. deepsignal2 call_mods
# CG
CUDA_VISIBLE_DEVICES=0 deepsignal2 call_mods --input_path fast5s/ --model_path model.dp2.CG.R9.4_1D.human_hx1.bn17_sn16.both_bilstm.b17_s16_epoch4.ckpt --result_file fast5s.CG.call_mods.tsv --corrected_group RawGenomeCorrected_000 --reference_path /path/to/genome/reference.fa --motifs CG --nproc 30 --nproc_gpu 6
python /path/to/deepsignal2/scripts/call_modification_frequency.py --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.tsv
```


## Usage
#### 1. Basecall and re-squiggle
Before run deepsignal, the raw reads should be basecalled ([Guppy>=3.6.1](https://nanoporetech.com/community)) and then be processed by the *re-squiggle* module of [tombo](https://github.com/nanoporetech/tombo).

Note:
- If the fast5 files are in multi-read FAST5 format, please use _multi_to_single_fast5_ command from the [ont_fast5_api package](https://github.com/nanoporetech/ont_fast5_api) to convert the fast5 files before using tombo (Ref to [issue #173](https://github.com/nanoporetech/tombo/issues/173) in [tombo](https://github.com/nanoporetech/tombo)).
```bash
multi_to_single_fast5 -i $multi_read_fast5_dir -s $single_read_fast5_dir -t 30 --recursive
```
- If the basecall results are saved as fastq, run the [*tombo proprecess annotate_raw_with_fastqs*](https://nanoporetech.github.io/tombo/resquiggle.html) command before *re-squiggle*.

For example:
```bash
# 1. basecall
guppy_basecaller -i fast5s/ -r -s fast5s_guppy --config dna_r9.4.1_450bps_hac_prom.cfg
# 2. proprecess fast5 if basecall results are saved in fastq format
cat fast5s_guppy/*.fastq > fast5s_guppy.fastq
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5s/ --fastq-filenames fast5s_guppy.fastq --basecall-group Basecall_1D_000 --basecall-subgroup BaseCalled_template --overwrite --processes 10
# 3. resquiggle, cmd: tombo resquiggle $fast5_dir $reference_fa
tombo resquiggle fast5s/ /path/to/genome/reference.fa --processes 10 --corrected-group RawGenomeCorrected_000 --basecall-group Basecall_1D_000 --overwrite
```

#### 2. extract features
Features of targeted sites can be extracted for training or testing.

For the example data (deepsignal2 extracts 17-mer-seq and 17*16-signal features of each CpG motif in reads by default. Note that the value of *--corrected_group* must be the same as that of *--corrected-group* in [tombo](https://github.com/nanoporetech/tombo).):
```bash
deepsignal2 extract -i fast5s --reference_path /path/to/genome/reference.fa -o fast5s.CG.features.tsv --corrected_group RawGenomeCorrected_000 --nproc 30 --motifs CG &
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

For example:
```bash
# call 5mCpGs for instance

# extracted-feature file as input
deepsignal2 call_mods --input_path fast5s.CG.features.tsv --model_path model.dp2.CG.R9.4_1D.human_hx1.bn17_sn16.both_bilstm.b17_s16_epoch4.ckpt --result_file fast5s.CG.call_mods.tsv --motifs CG --nproc 30 --nproc_gpu 6

# fast5 files as input, use CPU
CUDA_VISIBLE_DEVICES=-1 deepsignal2 call_mods --input_path fast5s/ --model_path model.dp2.CG.R9.4_1D.human_hx1.bn17_sn16.both_bilstm.b17_s16_epoch4.ckpt --result_file fast5s.CG.call_mods.tsv --corrected_group RawGenomeCorrected_000 --reference_path /path/to/genome/reference.fa --motifs CG --nproc 30
# fast5 files as input, use GPU
CUDA_VISIBLE_DEVICES=0 deepsignal2 call_mods --input_path fast5s/ --model_path model.dp2.CG.R9.4_1D.human_hx1.bn17_sn16.both_bilstm.b17_s16_epoch4.ckpt --result_file fast5s.CG.call_mods.tsv --corrected_group RawGenomeCorrected_000 --reference_path /path/to/genome/reference.fa --motifs CG --nproc 30 --nproc_gpu 6
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

A modification-frequency file can be generated by the script [scripts/call_modification_frequency.py](https://github.com/PengNi/deepsignal2/blob/master/scripts/call_modification_frequency.py) with the call_mods file as input:
```bash
# call 5mCpGs for instance

# output in tsv format
python /path/to/deepsignal2/scripts/call_modification_frequency.py --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.tsv
# output in bedMethyl format
python /path/to/deepsignal2/scripts/call_modification_frequency.py --input_path fast5s.CG.call_mods.tsv --result_file fast5s.CG.call_mods.frequency.bed --bed
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

#### 4. denoise training samples
```bash
# please use deepsignal2 denoise -h/--help for more details
deepsignal2 denoise --train_file /path/to/train/file
```

#### 5. train new models
A new model can be trained as follows:
```bash
# need to split training samples to two independent datasets for training and validating
# please use deepsignal2 train -h/--help for more details
deepsignal2 train --train_file /path/to/train/file --valid_file /path/to/valid/file --model_dir /dir/to/save/the/new/model
```


License
=========
Copyright (C) 2020 [Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Feng Luo](mailto:luofeng@clemson.edu), [Peng Ni](mailto:nipeng@csu.edu.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

[Jianxin Wang](mailto:jxwang@mail.csu.edu.cn), [Peng Ni](mailto:nipeng@csu.edu.cn),
School of Computer Science and Engineering, Central South University, Changsha 410083, China

[Feng Luo](mailto:luofeng@clemson.edu), School of Computing, Clemson University, Clemson, SC 29634, USA
