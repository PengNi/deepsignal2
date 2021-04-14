#!/usr/bin/env bash
# demo cmds for generating training samples
# 1. deepsignal2 extract (extract features from fast5s)
deepsignal2 extract --fast5_dir fast5s/ [--corrected_group --basecall_subgroup --reference_path] --methy_label 1 --motifs CG --mod_loc 0 --write_path samples_CG.hc_poses_positive.tsv [--nproc] --positions /path/to/file/contatining/high_confidence/positive/sites.tsv
deepsignal2 extract --fast5_dir fast5s/ [--corrected_group --basecall_subgroup --reference_path] --methy_label 0 --motifs CG --mod_loc 0 --write_path samples_CG.hc_poses_negative.tsv [--nproc] --positions /path/to/file/contatining/high_confidence/negative/sites.tsv

# 2. randomly select equally number (e.g., 10m) of positive and negative samples
# the selected positive and negative samples then can be combined and used for training, see step 4.
python /path/to/scripts/randsel_file_rows.py --ori_filepath samples_CG.hc_poses_positive.tsv --write_filepath samples_CG.hc_poses_positive.r10m.tsv --num_lines 10000000 --header false &
python /path/to/scripts/randsel_file_rows.py --ori_filepath samples_CG.hc_poses_negative.tsv --write_filepath samples_CG.hc_poses_negative.r10m.tsv --num_lines 10000000 --header false &

# 3. extract balanced negative (or positive) samples if needed
# for example, extract balanced negative samples of each kmer as the number of positive samples of the kmer
python /path/to/scripts/balance_samples_of_kmers.py --feafile samples_CG.hc_poses_negative.tsv --kmer_feafile samples_CG.hc_poses_positive.r10m.tsv --wfile samples_CG.hc_poses_negative.b10m.tsv

# 4. combine positive and negative samples for training
# after combining, the combined file can be splited into two files as training/validating set, see step 5.
python /path/to/scripts/concat_two_files.py --fp1 samples_CG.hc_poses_positive.r10m.tsv --fp2 samples_CG.hc_poses_negative.b10m.tsv --concated_fp samples_CG.hc_poses.rb20m.tsv

# 5. split samples for training/validating
# suppose file "samples_CG.hc_poses.rb20m" has 20000000 lines (samples), and we use 10k samples for validation
head -19990000 samples_CG.hc_poses.rb20m.tsv > samples_CG.hc_poses.rb20m.train.tsv
tail -10000 samples_CG.hc_poses.rb20m.tsv > samples_CG.hc_poses.rb20m.valid.tsv

# 6. train
CUDA_VISIBLE_DEVICES=0 deepsignal2 train --train_file samples_CG.hc_poses.rb20m.train.tsv --valid_file samples_CG.hc_poses.rb20m.valid.tsv --model_dir model.dplant.CG
