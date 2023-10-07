"""
call modifications from fast5 files or extracted features,
using tensorflow and the trained model.
output format: chromosome, pos, strand, pos_in_strand, read_name, read_loc,
prob_0, prob_1, called_label, seq
"""

from __future__ import absolute_import

import torch
import argparse
import os
import sys
import numpy as np
from sklearn import metrics

import gzip

# import multiprocessing as mp
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# TODO: when using below import, will raise AttributeError: 'Queue' object has no attribute '_size'
# TODO: didn't figure out why
# from .utils.process_utils import Queue
from torch.multiprocessing import Queue
import time

from .models import ModelBiLSTM
from .utils.process_utils import base2code_dna
from .utils.process_utils import code2base_dna
from .utils.process_utils import str2bool
from .utils.process_utils import display_args
from .utils.process_utils import nproc_to_call_mods_in_cpu_mode

from .extract_features import _extract_preprocess

from .utils.constants_torch import FloatTensor
from .utils.constants_torch import use_cuda

import uuid

from .extract_features import _get_read_sequened_strand
from .extract_features import get_aligner
from .extract_features import start_extract_processes
from .extract_features import start_map_threads
from .extract_features import _reads_processed_stats

from .utils.process_utils import get_logger
LOGGER = get_logger(__name__)

# add this export temporarily
os.environ['MKL_THREADING_LAYER'] = 'GNU'

queue_size_border = 2000
queue_size_border_f5batch = 100
time_wait = 3


def _read_features_file(features_file, features_batch_q, f5_batch_size=10):
    LOGGER.info("read_features process-{} starts".format(os.getpid()))
    r_num, b_num = 0, 0

    if features_file.endswith(".gz"):
        infile = gzip.open(features_file, 'rt')
    else:
        infile = open(features_file, 'r')

    sampleinfo = []  # contains: chromosome, pos, strand, pos_in_strand, read_name, read_strand
    kmers = []
    base_means = []
    base_stds = []
    base_signal_lens = []
    base_probs = []
    k_signals = []
    labels = []

    line = next(infile)
    words = line.strip().split("\t")
    readid_pre = words[4]

    sampleinfo.append("\t".join(words[0:6]))
    kmers.append([base2code_dna[x] for x in words[6]])
    base_means.append([float(x) for x in words[7].split(",")])
    base_stds.append([float(x) for x in words[8].split(",")])
    base_signal_lens.append([int(x) for x in words[9].split(",")])
    base_probs.append(np.zeros(13))#[float(x) for x in words[10].split(",")])
    k_signals.append([[float(y) for y in x.split(",")] for x in words[11].split(";")])
    labels.append(int(words[12]))

    for line in infile:
        words = line.strip().split("\t")
        readidtmp = words[4]
        if readidtmp != readid_pre:
            r_num += 1
            readid_pre = readidtmp
            if r_num % f5_batch_size == 0:
                features_batch_q.put((sampleinfo, kmers, base_means, base_stds,
                                      base_signal_lens, base_probs, k_signals, labels))
                while features_batch_q.qsize() > queue_size_border_f5batch:
                    time.sleep(time_wait)
                sampleinfo = []
                kmers = []
                base_means = []
                base_stds = []
                base_signal_lens = []
                base_probs = []
                k_signals = []
                labels = []
                b_num += 1

        sampleinfo.append("\t".join(words[0:6]))
        kmers.append([base2code_dna[x] for x in words[6]])
        base_means.append([float(x) for x in words[7].split(",")])
        base_stds.append([float(x) for x in words[8].split(",")])
        base_signal_lens.append([int(x) for x in words[9].split(",")])
        base_probs.append(np.zeros(13))#[float(x) for x in words[10].split(",")])
        k_signals.append([[float(y) for y in x.split(",")] for x in words[11].split(";")])
        labels.append(int(words[12]))
    infile.close()
    r_num += 1
    if len(sampleinfo) > 0:
        features_batch_q.put((sampleinfo, kmers, base_means, base_stds,
                              base_signal_lens, base_probs, k_signals, labels))
        b_num += 1
    features_batch_q.put("kill")
    LOGGER.info("read_features process-{} ending, "
                "read {} reads in {} f5-batches({})".format(os.getpid(), r_num, b_num, f5_batch_size))


def _call_mods(features_batch, model, batch_size, device=0):
    # features_batch: 1. if from _read_features_file(), has 1 * args.batch_size samples (not any more, modified)
    # --------------: 2. if from _read_features_from_fast5s(), has uncertain number of samples
    sampleinfo, kmers, base_means, base_stds, base_signal_lens, base_probs, \
        k_signals, labels = features_batch
    labels = np.reshape(labels, (len(labels)))

    pred_str = []
    accuracys = []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]
        b_kmers = kmers[batch_s:batch_e]
        b_base_means = base_means[batch_s:batch_e]
        b_base_stds = base_stds[batch_s:batch_e]
        b_base_signal_lens = base_signal_lens[batch_s:batch_e]
        b_base_probs = base_probs[batch_s:batch_e]
        b_k_signals = k_signals[batch_s:batch_e]
        b_labels = labels[batch_s:batch_e]
        if len(b_sampleinfo) > 0:
            voutputs, vlogits = model(FloatTensor(b_kmers, device), FloatTensor(b_base_means, device),
                                      FloatTensor(b_base_stds, device), FloatTensor(b_base_signal_lens, device),
                                      FloatTensor(b_base_probs, device), FloatTensor(b_k_signals, device))
            _, vpredicted = torch.max(vlogits.data, 1)
            if use_cuda:
                vlogits = vlogits.cpu()
                vpredicted = vpredicted.cpu()

            predicted = vpredicted.numpy()
            logits = vlogits.data.numpy()

            acc_batch = metrics.accuracy_score(
                y_true=b_labels, y_pred=predicted)
            accuracys.append(acc_batch)

            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, pos_in_strand, read_name, read_strand, prob_0, prob_1, called_label, seq
                prob_0, prob_1 = logits[idx][0], logits[idx][1]
                prob_0_norm = round(prob_0 / (prob_0 + prob_1), 6)
                prob_1_norm = round(1 - prob_0_norm, 6)
                # kmer-5
                b_idx_kmer = ''.join([code2base_dna[x] for x in b_kmers[idx]])
                center_idx = int(np.floor(len(b_idx_kmer) / 2))
                bkmer_start = center_idx - 2 if center_idx - 2 >= 0 else 0
                bkmer_end = center_idx + 3 if center_idx + 3 <= len(b_idx_kmer) else len(b_idx_kmer)

                pred_str.append("\t".join([b_sampleinfo[idx], str(prob_0_norm),
                                           str(prob_1_norm), str(predicted[idx]),
                                           b_idx_kmer[bkmer_start:bkmer_end]]))
            batch_num += 1
    accuracy = np.mean(accuracys) if len(accuracys) > 0 else 0

    return pred_str, accuracy, batch_num


def _call_mods_q(model_path, features_batch_q, pred_str_q, success_file, args, device=0):
    LOGGER.info('call_mods process-{} starts'.format(os.getpid()))
    model = ModelBiLSTM(args.seq_len, args.signal_len, args.layernum1, args.layernum2, args.class_num,
                        args.dropout_rate, args.hid_rnn,
                        args.n_vocab, args.n_embed, str2bool(args.is_base), str2bool(args.is_signallen),
                        str2bool(args.is_trace),
                        args.model_type, device=device)

    try:
        para_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception:
        para_dict = torch.jit.load(model_path)
    # para_dict = torch.load(model_path, map_location=torch.device(device))
    model_dict = model.state_dict()
    model_dict.update(para_dict)
    model.load_state_dict(model_dict)
    del model_dict

    if use_cuda:
        model = model.cuda(device)
    model.eval()

    accuracy_list = []
    batch_num_total = 0
    while True:
        # if os.path.exists(success_file):
        #     break

        if features_batch_q.empty():
            time.sleep(time_wait)
            continue

        features_batch = features_batch_q.get()
        if features_batch == "kill":
            # deprecate successfile, use "kill" signal multi times to kill each process
            features_batch_q.put("kill")
            # open(success_file, 'w').close()
            break

        pred_str, accuracy, batch_num = _call_mods(features_batch, model, args.batch_size, device)

        pred_str_q.put(pred_str)
        while pred_str_q.qsize() > queue_size_border:
            time.sleep(time_wait)
        # for debug
        # print("call_mods process-{} reads 1 batch, features_batch_q:{}, "
        #       "pred_str_q: {}".format(os.getpid(), features_batch_q.qsize(), pred_str_q.qsize()))
        accuracy_list.append(accuracy)
        batch_num_total += batch_num
    # print('total accuracy in process {}: {}'.format(os.getpid(), np.mean(accuracy_list)))
    LOGGER.info('call_mods process-{} ending, proceed {} batches({})'.format(os.getpid(), batch_num_total,
                                                                             args.batch_size))


def _write_predstr_to_file(write_fp, predstr_q):
    LOGGER.info("write_process-{} starts".format(os.getpid()))
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep()
            if predstr_q.empty():
                time.sleep(time_wait)
                continue
            pred_str = predstr_q.get()
            if pred_str == "kill":
                LOGGER.info("write_process-{} finished".format(os.getpid()))
                break
            for one_pred_str in pred_str:
                wf.write(one_pred_str + "\n")
            wf.flush()


def _call_mods_from_fast5s_gpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, chrom2seqs,
                               model_path, success_file, read_strand,
                               args):
    features_batch_q = Queue()
    error_q = Queue()
    pred_str_q = Queue()

    if args.mapping:
        aligner = get_aligner(ref_path, args.best_n)

    nproc = args.nproc
    nproc_gpu = args.nproc_gpu
    if nproc_gpu < 1:
        nproc_gpu = 1
    if nproc <= nproc_gpu + 1:
        LOGGER.info("--nproc must be >= --nproc_gpu + 2!!")
        nproc = nproc_gpu + 1 + 1

    fast5s_q.put("kill")
    nproc_extr = nproc - nproc_gpu - 1

    extract_ps, map_conns = start_extract_processes(fast5s_q, features_batch_q, error_q, nproc_extr,
                                                    args.basecall_group, args.basecall_subgroup,
                                                    args.normalize_method,
                                                    args.mapq, args.identity, args.coverage_ratio,
                                                    motif_seqs, chrom2len, positions, read_strand,
                                                    args.mod_loc, args.seq_len, args.signal_len, args.methy_label,
                                                    args.pad_only_r,  args.single, args.f5_batch_size,
                                                    args.mapping, args.corrected_group, chrom2seqs,
                                                    is_to_str=False, is_batchlize=True,is_trace=args.trace)
    call_mods_gpu_procs = []
    gpulist = _get_gpus()
    gpuindex = 0
    for i in range(nproc_gpu):
        p_call_mods_gpu = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q,
                                                                success_file, args, gpulist[gpuindex]),
                                     name="caller_{:03d}".format(i))
        gpuindex += 1
        p_call_mods_gpu.daemon = True
        p_call_mods_gpu.start()
        call_mods_gpu_procs.append(p_call_mods_gpu)

    # print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q),
                     name="writer")
    p_w.daemon = True
    p_w.start()

    if args.mapping:
        map_read_ts = start_map_threads(map_conns, aligner)

    # finish processes
    error2num = {-1: 0, -2: 0, -3: 0, 0: 0}  # (-1, -2, -3, 0)
    while True:
        running = any(p.is_alive() for p in extract_ps)
        while not error_q.empty():
            error2numtmp = error_q.get()
            for ecode in error2numtmp.keys():
                error2num[ecode] += error2numtmp[ecode]
        if not running:
            break

    for p in extract_ps:
        p.join()
    if args.mapping:
        for map_t in map_read_ts:
            map_t.join()
    features_batch_q.put("kill")

    for p_call_mods_gpu in call_mods_gpu_procs:
        p_call_mods_gpu.join()
    pred_str_q.put("kill")

    p_w.join()

    _reads_processed_stats(error2num, len_fast5s, args.single)


def _call_mods_from_fast5s_cpu2(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, chrom2seqs,
                                model_path, success_file, read_strand,
                                args):
    features_batch_q = Queue()
    error_q = Queue()
    pred_str_q = Queue()

    if args.mapping:
        aligner = get_aligner(ref_path, args.best_n)

    nproc = args.nproc
    nproc_call_mods = nproc_to_call_mods_in_cpu_mode
    if nproc <= nproc_call_mods + 1:
        nproc = nproc_call_mods + 1 + 1

    fast5s_q.put("kill")
    nproc_extr = nproc - nproc_call_mods - 1
    extract_ps, map_conns = start_extract_processes(fast5s_q, features_batch_q, error_q, nproc_extr,
                                                    args.basecall_group, args.basecall_subgroup,
                                                    args.normalize_method,
                                                    args.mapq, args.identity, args.coverage_ratio,
                                                    motif_seqs, chrom2len, positions, read_strand,
                                                    args.mod_loc, args.seq_len, args.signal_len, args.methy_label,
                                                    args.pad_only_r, args.single, args.f5_batch_size,
                                                    args.mapping, args.corrected_group, chrom2seqs,
                                                    is_to_str=False, is_batchlize=True,is_trace=args.trace)

    call_mods_procs = []
    for i in range(nproc_call_mods):
        p_call_mods = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q,
                                                            success_file, args),
                                 name="caller_{:03d}".format(i))
        p_call_mods.daemon = True
        p_call_mods.start()
        call_mods_procs.append(p_call_mods)

    # print("write_process started..")
    p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q),
                     name="writer")
    p_w.daemon = True
    p_w.start()

    if args.mapping:
        map_read_ts = start_map_threads(map_conns, aligner)

    # finish processes
    error2num = {-1: 0, -2: 0, -3: 0, 0: 0}  # (-1, -2, -3, 0)
    while True:
        running = any(p.is_alive() for p in extract_ps)
        while not error_q.empty():
            error2numtmp = error_q.get()
            for ecode in error2numtmp.keys():
                error2num[ecode] += error2numtmp[ecode]
        if not running:
            break

    for p in extract_ps:
        p.join()
    if args.mapping:
        for map_t in map_read_ts:
            map_t.join()
    features_batch_q.put("kill")

    for p_call_mods in call_mods_procs:
        p_call_mods.join()
    pred_str_q.put("kill")

    p_w.join()

    _reads_processed_stats(error2num, len_fast5s, args.single)


def _get_gpus():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpulist = list(range(num_gpus))
    else:
        gpulist = [0]
    return gpulist * 1000


def call_mods(args):
    start = time.time()
    LOGGER.info("[call_mods] starts")
    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        raise ValueError("--model_path is not set right!")
    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        raise ValueError("--input_path does not exist!")
    success_file = input_path.rstrip("/") + "." + str(uuid.uuid1()) + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)

    if os.path.isdir(input_path):
        if args.reference_path is None:
            raise ValueError("--reference_path is required to be set!")
        ref_path = os.path.abspath(args.reference_path)
        if not os.path.exists(ref_path):
            raise ValueError("--reference_path is not set right!")

        is_recursive = str2bool(args.recursively)
        is_dna = False if args.rna else True
        motif_seqs, chrom2len, fast5s_q, len_fast5s, \
            positions, contigs = _extract_preprocess(input_path, is_recursive,
                                                     args.motifs, is_dna, ref_path,
                                                     args.f5_batch_size, args.positions,
                                                     args)
        read_strand = _get_read_sequened_strand(args.basecall_subgroup)

        if use_cuda:
            _call_mods_from_fast5s_gpu(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs,
                                       model_path, success_file, read_strand, args)
        else:
            _call_mods_from_fast5s_cpu2(ref_path, motif_seqs, chrom2len, fast5s_q, len_fast5s, positions, contigs,
                                        model_path, success_file, read_strand, args)
    else:
        features_batch_q = Queue()
        p_rf = mp.Process(target=_read_features_file, args=(input_path, features_batch_q,
                                                            args.f5_batch_size),
                          name="reader")
        p_rf.daemon = True
        p_rf.start()

        pred_str_q = Queue()

        predstr_procs = []

        if use_cuda:
            nproc_dp = args.nproc_gpu
            if nproc_dp < 1:
                nproc_dp = 1
        else:
            nproc = args.nproc
            if nproc < 3:
                LOGGER.info("--nproc must be >= 3!!")
                nproc = 3
            nproc_dp = nproc - 2
            if nproc_dp > nproc_to_call_mods_in_cpu_mode:
                nproc_dp = nproc_to_call_mods_in_cpu_mode

        gpulist = _get_gpus()
        gpuindex = 0
        for i in range(nproc_dp):
            p = mp.Process(target=_call_mods_q, args=(model_path, features_batch_q, pred_str_q,
                                                      success_file, args, gpulist[gpuindex]),
                           name="caller_{:03d}".format(i))
            gpuindex += 1
            p.daemon = True
            p.start()
            predstr_procs.append(p)

        # print("write_process started..")
        p_w = mp.Process(target=_write_predstr_to_file, args=(args.result_file, pred_str_q),
                         name="writer")
        p_w.daemon = True
        p_w.start()

        for p in predstr_procs:
            p.join()

        # print("finishing the write_process..")
        pred_str_q.put("kill")

        p_rf.join()

        p_w.join()

    if os.path.exists(success_file):
        os.remove(success_file)
    LOGGER.info("[call_mods] costs %.2f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser("call modifications")

    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("--input_path", "-i", action="store", type=str,
                         required=True,
                         help="the input path, can be a signal_feature file from extract_features.py, "
                              "or a directory of fast5 files. If a directory of fast5 files is provided, "
                              "args in FAST5_EXTRACTION and MAPPING should (reference_path must) be provided.")
    p_input.add_argument("--f5_batch_size", action="store", type=int, default=50,
                         required=False,
                         help="number of files to be processed by each process one time, default 50")

    p_call = parser.add_argument_group("CALL")
    p_call.add_argument("--model_path", "-m", action="store", type=str, required=True,
                        help="file path of the trained model (.ckpt)")

    # model input
    p_call.add_argument('--model_type', type=str, default="both_bilstm",
                        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                        required=False,
                        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                             "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    p_call.add_argument('--seq_len', type=int, default=13, required=False,
                        help="len of kmer. default 13")
    p_call.add_argument('--signal_len', type=int, default=15, required=False,
                        help="signal num of one base, default 15")

    # model param
    p_call.add_argument('--layernum1', type=int, default=3,
                        required=False, help="lstm layer num for combined feature, default 3")
    p_call.add_argument('--layernum2', type=int, default=1,
                        required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0, required=False)
    p_call.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    p_call.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")
    p_call.add_argument('--is_base', type=str, default="yes", required=False,
                        help="is using base features in seq model, default yes")
    p_call.add_argument('--is_signallen', type=str, default="yes", required=False,
                        help="is using signal length feature of each base in seq model, default yes")
    p_call.add_argument('--is_trace', type=str, default="no", required=False,
                        help="is using trace (base prob) feature of each base in seq model, default yes")

    p_call.add_argument("--batch_size", "-b", default=512, type=int, required=False,
                        action="store", help="batch size, default 512")

    # BiLSTM model param
    p_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiLSTM hidden_size for combined feature")

    p_output = parser.add_argument_group("OUTPUT")
    p_output.add_argument("--result_file", "-o", action="store", type=str, required=True,
                          help="the file path to save the predicted result")

    p_f5 = parser.add_argument_group("FAST5_EXTRACTION")
    p_f5.add_argument("--single", action="store_true", default=False, required=False,
                      help='the fast5 files are in single-read format')
    p_f5.add_argument("--recursively", "-r", action="store", type=str, required=False,
                      default='yes', help='is to find fast5 files from fast5 dir recursively. '
                                          'default true, t, yes, 1')
    p_f5.add_argument("--rna", action="store_true", default=False, required=False,
                      help='the fast5 files are from RNA samples. if is rna, the signals are reversed. '
                           'NOTE: Currently no use, waiting for further extentsion')
    p_f5.add_argument("--basecall_group", action="store", type=str, required=False,
                      default=None,
                      help='basecall group generated by Guppy. e.g., Basecall_1D_000')
    p_f5.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                      default='BaseCalled_template',
                      help='the basecall subgroup of fast5 files. default BaseCalled_template')
    p_f5.add_argument("--reference_path", action="store",
                      type=str, required=False,
                      help="the reference file to be used, usually is a .fa file")
    p_f5.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                      default="mad", required=False,
                      help="the way for normalizing signals in read level. "
                           "mad or zscore, default mad")
    p_f5.add_argument("--methy_label", action="store", type=int,
                      choices=[1, 0], required=False, default=1,
                      help="the label of the interested modified bases, this is for training."
                           " 0 or 1, default 1")
    p_f5.add_argument("--motifs", action="store", type=str,
                      required=False, default='CG',
                      help='motif seq to be extracted, default: CG. '
                           'can be multi motifs splited by comma '
                           '(no space allowed in the input str), '
                           'or use IUPAC alphabet, '
                           'the mod_loc of all motifs must be '
                           'the same')
    p_f5.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                      help='0-based location of the targeted base in the motif, default 0')
    p_f5.add_argument("--pad_only_r", action="store_true", default=False,
                      help="pad zeros to only the right of signals array of one base, "
                           "when the number of signals is less than --signal_len. "
                           "default False (pad in two sides).")
    p_f5.add_argument("--positions", action="store", type=str,
                      required=False, default=None,
                      help="file with a list of positions interested (must be formatted as tab-separated file"
                           " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                           "need to be set. --positions is used to narrow down the range of the trageted "
                           "motif locs. default None")
    p_f5.add_argument("--trace", action="store_true", default=False, required=False,
                       help='use trace, default false')
    p_mape = parser.add_argument_group("MAPe")
    p_mape.add_argument("--corrected_group", action="store", type=str, required=False,
                        default='RawGenomeCorrected_000',
                        help='the corrected_group of fast5 files, '
                             'default RawGenomeCorrected_000')

    p_mapping = parser.add_argument_group("MAPPING")
    p_mapping.add_argument("--mapping", action="store_true", default=False, required=False,
                           help='use MAPPING to get alignment, default false')
    p_mapping.add_argument("--mapq", type=int, default=10, required=False,
                           help="MAPping Quality cutoff for selecting alignment items, default 10")
    p_mapping.add_argument("--identity", type=float, default=0.75, required=False,
                           help="identity cutoff for selecting alignment items, default 0.75")
    p_mapping.add_argument("--coverage_ratio", type=float, default=0.75, required=False,
                           help="percent of coverage, read alignment len against read len, default 0.75")
    p_mapping.add_argument("--best_n", "-n", type=int, default=1, required=False,
                           help="best_n arg in mappy(minimap2), default 1")

    parser.add_argument("--nproc", "-p", action="store", type=int, default=10,
                        required=False, help="number of processes to be used, default 10.")
    parser.add_argument("--nproc_gpu", action="store", type=int, default=2,
                        required=False, help="number of processes to use gpu (if gpu is available), "
                                             "1 or a number less than nproc-1, no more than "
                                             "nproc/4 is suggested. default 2.")

    args = parser.parse_args()
    display_args(args)
    call_mods(args)


if __name__ == '__main__':
    sys.exit(main())
