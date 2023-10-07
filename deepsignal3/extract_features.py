"""the feature extraction module.
output format:
chrom, pos, alignstrand, pos_in_strand, readname, read_loc, k_mer, signal_means,
signal_stds, signal_lens, base_probs, raw_signals, methy_label
"""
# TODO: use pyguppyclient (https://github.com/nanoporetech/pyguppyclient) instead of --fast5-out

from __future__ import absolute_import

import sys
import os
import argparse
import time
# import h5py
import random
import numpy as np
import multiprocessing as mp
# TODO: when using below import, will raise AttributeError: 'Queue' object has no attribute '_size'
# TODO: in call_mods module, didn't figure out why
# from .utils.process_utils import Queue
from multiprocessing import Queue
from statsmodels import robust

from .utils.process_utils import str2bool
from .utils.process_utils import display_args
from .utils.process_utils import get_fast5s
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import complement_seq

from .utils.ref_reader import get_contig2len
from .utils.ref_reader import get_contig2len_n_seq

from .utils import fast5_reader
from .utils.process_utils import base2code_dna

# from .utils.process_utils import generate_aligner_with_options

import re
from .utils.process_utils import CIGAR_REGEX
from .utils.process_utils import CIGAR2CODE

import mappy
import threading
from collections import namedtuple
from .utils.process_utils import get_logger
LOGGER = get_logger(__name__)

queue_size_border = 2000
time_wait = 3

key_sep = "||"

MAP_RES = namedtuple('MAP_RES', (
    'read_id', 'q_seq', 'ref_seq', 'ctg', 'strand', 'r_st', 'r_en',
    'q_st', 'q_en', 'cigar', 'mapq'))


# extract readseq with mapped signals ===================================================
def _group_signals_by_movetable_v2(trimed_signals, movetable, stride):
    """

    :param trimed_signals:
    :param movetable: numpy array
    :param stride:
    :return:
    """
    assert movetable[0] == 1
    # TODO: signal_duration not exactly equal to call_events * stride
    # TODO: maybe this move * stride way is not right!
    assert len(trimed_signals) >= len(movetable) * stride
    signal_group = []
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    for move_idx in range(len(move_pos) - 1):
        start, end = move_pos[move_idx], move_pos[move_idx + 1]
        signal_group.append(trimed_signals[(start * stride):(end * stride)].tolist())
    assert len(signal_group) == np.sum(movetable)
    return signal_group


def _get_base_prob_from_tracetable(tracetable, movetable):
    if tracetable is None:
        return []
    assert movetable[0] == 1
    assert len(movetable) == len(tracetable)
    base_probs = []
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    for move_idx in range(len(move_pos) - 1):
        start, end = move_pos[move_idx], move_pos[move_idx + 1]
        prob_col = np.sum(tracetable[start:end, :], axis=0)
        prob_all = np.sum(prob_col)
        base_probs.append(prob_col / prob_all)
    return base_probs


def _normalize_signals(signals, normalize_method="mad"):
    if normalize_method == 'zscore':
        sshift, sscale = np.mean(signals), float(np.std(signals))
    elif normalize_method == 'mad':
        sshift, sscale = np.median(signals), float(robust.mad(signals))
    else:
        raise ValueError("")
    if sscale == 0.0:
        norm_signals = signals
    else:
        norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def _get_read_sequened_strand(basecall_subgroup):
    if basecall_subgroup.endswith('template'):
        strand = 't'
    else:
        strand = 'c'
    return strand


# Move table:
# https://github.com/adnaniazi/tailfindr/blob/master/R/extract-read-data.R
# https://github.com/Fabianexe/SlaPPy/blob/master/slappy/fast5/__init__.py
# TODO: consider RNA mode, raw signals need to be reversed?
def _get_read_signalinfo(fast5_fn, basecall_group, basecall_subgroup, norm_method,
                         is_single=False, readname="", is_mapping=False, corrected_group="",is_trace=False):
    """

    :param fast5_fn:
    :param basecall_group:
    :param basecall_subgroup:
    :param norm_method:
    :param readname: if is_single=False, readname must not be ""
    :param is_single: is the fast5_fn is in single-read format
    :return: success(0/1), (readid, baseseq, signal_group)
    """
    try:
        success = 1

        fast5read = fast5_reader.SingleFast5(fast5_fn, is_single, readname)
        #print('1: ')
        readid = fast5read.get_readid()
        bgrp = basecall_group if basecall_group is not None else fast5read.get_lastest_basecallgroup()
        baseseq = fast5read.get_seq(bgrp, basecall_subgroup)
        movetable = fast5read.get_move(bgrp, basecall_subgroup)
        signal_stride = fast5read.get_stride(bgrp)
        #print('2: ')
        rawsignals = fast5read.get_raw_signal()
        rawsignals = fast5read.rescale_signals(rawsignals)
        #print('3: ')
        seggrp = fast5read.get_basecallgroup_related_sementation(bgrp)
        signal_start, signal_duration = fast5read.get_segmentation_start(seggrp), \
            fast5read.get_segmentation_duration(seggrp)
        #print('4: ')
        tracetable = None
        if is_trace:
            tracetable = fast5read.get_trace_4(bgrp, basecall_subgroup)
        #print('5: ')
        fast5read.check_fastq_seqlen(bgrp, basecall_subgroup)
        fast5read.check_signallen_against_segmentation(seggrp)

        mapinfo = None
        if not is_mapping:
            mapinfo = fast5read.get_map_info(corrected_group, basecall_subgroup)

        fast5read.close()

        # map signal to base
        mapsignals = _normalize_signals(rawsignals[signal_start:(signal_start + signal_duration)],
                                        norm_method)
        signal_group = _group_signals_by_movetable_v2(mapsignals, movetable, signal_stride)

        base_probs = _get_base_prob_from_tracetable(tracetable, movetable)

        return success, (readid, baseseq, signal_group, base_probs, mapinfo)
    except IOError:
        LOGGER.error('Error in reading file-{}, skipping'.format(fast5_fn))
        return 0, None
    except AssertionError:
        LOGGER.error('Error in mapping signal2base, file-{}, skipping'.format(fast5_fn))
        return 0, None
    except KeyError:
        LOGGER.error('Error in getting group from file-{}, skipping'.format(fast5_fn))
        return 0, None
# =======================================================================================


# map signals to refseq =================================================================
# from megalodon.mapping
def get_aligner(ref_path, best_n):
    LOGGER.info("get mappy(minimap2) Aligner")
    aligner = mappy.Aligner(str(ref_path),
                            preset=str("map-ont"),
                            best_n=best_n)
    return aligner


def align_read(q_seq, aligner, map_thr_buf, read_id=None):
    try:
        # enumerate all alignments to avoid memory leak from mappy
        r_algn = list(aligner.map(str(q_seq), buf=map_thr_buf))[0]
    except IndexError:
        # alignment not produced
        return None

    ref_seq = str(aligner.seq(r_algn.ctg, r_algn.r_st, r_algn.r_en)).upper()
    if r_algn.strand == -1:
        try:
            ref_seq = complement_seq(ref_seq)
        except KeyError:
            LOGGER.debug("ref_seq contions U base: {}".format(ref_seq))
            ref_seq = complement_seq(ref_seq, "RNA")
    # coord 0-based
    return MAP_RES(
        read_id=read_id, q_seq=str(q_seq).upper(), ref_seq=ref_seq, ctg=r_algn.ctg,
        strand=r_algn.strand, r_st=r_algn.r_st, r_en=r_algn.r_en,
        q_st=r_algn.q_st, q_en=r_algn.q_en, cigar=r_algn.cigar,
        mapq=r_algn.mapq)


# from megalodon.mapping
def _compute_pct_identity(cigar):
    nalign, nmatch = 0, 0
    for op_len, op in cigar:
        if op not in (4, 5):
            nalign += op_len
        if op in (0, 7):
            nmatch += op_len
    return nmatch / float(nalign)


# from megalodon.mapping
def parse_cigar(r_cigar, strand, ref_len):
    fill_invalid = -1
    # get each base calls genomic position
    r_to_q_poss = np.full(ref_len + 1, fill_invalid, dtype=np.int32)
    # process cigar ops in read direction
    curr_r_pos, curr_q_pos = 0, 0
    cigar_ops = r_cigar if strand == 1 else r_cigar[::-1]
    for op_len, op in cigar_ops:
        if op == 1:
            # inserted bases into ref
            curr_q_pos += op_len
        elif op in (2, 3):
            # deleted ref bases
            for r_pos in range(curr_r_pos, curr_r_pos + op_len):
                r_to_q_poss[r_pos] = curr_q_pos
            curr_r_pos += op_len
        elif op in (0, 7, 8):
            # aligned bases
            for op_offset in range(op_len):
                r_to_q_poss[curr_r_pos + op_offset] = curr_q_pos + op_offset
            curr_q_pos += op_len
            curr_r_pos += op_len
        elif op == 6:
            # padding (shouldn't happen in mappy)
            pass
    r_to_q_poss[curr_r_pos] = curr_q_pos
    if r_to_q_poss[-1] == fill_invalid:
        raise ValueError((
            'Invalid cigar string encountered. Reference length: {}  Cigar ' +
            'implied reference length: {}').format(ref_len, curr_r_pos))

    return r_to_q_poss


def _map_read_to_ref_process(aligner, map_conn):
    map_thr_buf = mappy.ThreadBuffer()
    LOGGER.debug("align process starts")
    while True:
        try:
            read_id, q_seq = map_conn.recv()
        except EOFError:
            LOGGER.debug("align process ending")
            break
        map_res = align_read(q_seq, aligner, map_thr_buf, read_id)
        if map_res is None:
            map_conn.send((0, None))
        else:
            map_res = tuple(map_res)
            map_conn.send((1, map_res))
# =======================================================================================


# =======extract mapinfo from fast5 =====================================================
def _convert_cigarstring2tuple(cigarstr):
    # tuple in (oplen, op) format, like 30M -> (30, 0)
    return [(int(m[0]), CIGAR2CODE[m[-1]]) for m in CIGAR_REGEX.findall(cigarstr)]
# =======================================================================================


# extract signals of kmers ==============================================================
def _get_signals_rect(signals_list, signals_len=16, pad_only_r=False):
    signals_rect = []
    for signals_tmp in signals_list:
        signals = list(np.around(signals_tmp, decimals=6))
        if len(signals) < signals_len:
            pad0_len = signals_len - len(signals)
            if not pad_only_r:
                pad0_left = pad0_len // 2
                pad0_right = pad0_len - pad0_left
                signals = [0.] * pad0_left + signals + [0.] * pad0_right
            else:
                signals = signals + [0.] * pad0_len
        elif len(signals) > signals_len:
            signals = [signals[x] for x in sorted(random.sample(range(len(signals)),
                                                                signals_len))]
        signals_rect.append(signals)
    return signals_rect


def _extract_features(ref_mapinfo, motif_seqs, chrom2len, positions, read_strand,
                      mod_loc, seq_len, signal_len, methy_label,
                      pad_only_r):
    # read_id, ref_seq, chrom, strand, r_st, r_en, ref_signal_grp, ref_baseprobs = ref_mapinfo
    # strand = "+" if strand == 1 else "-"

    # seg_mapping: (len(ref_seq), item_chrom, strand_code, item_ref_s, item_ref_e)
    read_id, ref_seq, ref_readlocs, ref_signal_grp, ref_baseprobs, seg_mapping = ref_mapinfo

    if seq_len % 2 == 0:
        raise ValueError("--seq_len must be odd")
    num_bases = (seq_len - 1) // 2

    feature_lists = []
    # WARN: cannot make sure the ref_offsets are all targeted motifs in corresponding read,
    # WARN: cause there may be mismatches/indels in those postions.
    # WARN: see also parse_cigar()/_process_read_map()/_process_read_nomap()
    ref_offsets = get_refloc_of_methysite_in_motif(ref_seq, set(motif_seqs), mod_loc)
    for off_loc in ref_offsets:
        if num_bases <= off_loc < len(ref_seq) - num_bases:
            chrom, strand, r_st, r_en = None, None, None, None
            seg_off_loc = None
            seg_len_accum = 0
            for seg_tmp in seg_mapping:
                seg_len_accum += seg_tmp[0]
                if off_loc < seg_len_accum:
                    seg_off_loc = off_loc - (seg_len_accum - seg_tmp[0])
                    chrom, strand, r_st, r_en = seg_tmp[1:]
                    strand = "+" if strand == 1 else "-"
                    break

            abs_loc = r_st + seg_off_loc if strand == "+" else r_en - 1 - seg_off_loc
            read_loc = ref_readlocs[off_loc]

            if (positions is not None) and (key_sep.join([chrom, str(abs_loc), strand]) not in positions):
                continue

            loc_in_strand = abs_loc if (strand == "+" or chrom == "read") else chrom2len[chrom] - 1 - abs_loc
            k_mer = ref_seq[(off_loc - num_bases):(off_loc + num_bases + 1)]
            k_signals = ref_signal_grp[(off_loc - num_bases):(off_loc + num_bases + 1)]
            
            k_baseprobs = ref_baseprobs[(off_loc - num_bases):(off_loc + num_bases + 1)] if len(ref_baseprobs)>0 else np.zero(seq_len)

            signal_lens = [len(x) for x in k_signals]

            signal_means = [np.mean(x) for x in k_signals]
            signal_stds = [np.std(x) for x in k_signals]

            k_signals_rect = _get_signals_rect(k_signals, signal_len, pad_only_r)

            feature_lists.append((chrom, abs_loc, strand, loc_in_strand, read_id, read_loc,
                                  k_mer, signal_means, signal_stds, signal_lens,
                                  k_baseprobs,
                                  k_signals_rect, methy_label))
    return feature_lists
# =======================================================================================


# write =================================================================================
def _features_to_str(features):
    """
    :param features: a tuple
    :return:
    """
    chrom, pos, alignstrand, loc_in_ref, readname, read_loc, k_mer, signal_means, signal_stds, \
        signal_lens, \
        k_baseprobs, \
        k_signals_rect, methy_label = features
    means_text = ','.join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ','.join([str(x) for x in np.around(signal_stds, decimals=6)])
    signal_len_text = ','.join([str(x) for x in signal_lens])
    base_probs_text = ",".join([str(x) for x in np.around(k_baseprobs, decimals=6)])
    k_signals_text = ';'.join([",".join([str(y) for y in x]) for x in k_signals_rect])

    return "\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname, str(read_loc), k_mer, means_text,
                      stds_text, signal_len_text, base_probs_text, k_signals_text, str(methy_label)])


def _write_featurestr_to_file(write_fp, featurestr_q):
    LOGGER.info('write_process-{} starts'.format(os.getpid()))
    with open(write_fp, 'w') as wf:
        while True:
            # during test, it's ok without the sleep(time_wait)
            if featurestr_q.empty():
                time.sleep(time_wait)
                continue
            features_str = featurestr_q.get()
            if features_str == "kill":
                LOGGER.info('write_process-{} finished'.format(os.getpid()))
                break
            for one_features_str in features_str:
                wf.write(one_features_str + "\n")
            wf.flush()


def _write_featurestr_to_dir(write_dir, featurestr_q, w_batch_num):
    LOGGER.info('write_process-{} starts'.format(os.getpid()))
    if os.path.exists(write_dir):
        if os.path.isfile(write_dir):
            raise FileExistsError("{} already exists as a file, please use another write_dir".format(write_dir))
    else:
        os.makedirs(write_dir)

    file_count = 0
    wf = open("/".join([write_dir, str(file_count) + ".tsv"]), "w")
    batch_count = 0
    while True:
        # during test, it's ok without the sleep(time_wait)
        if featurestr_q.empty():
            time.sleep(time_wait)
            continue
        features_str = featurestr_q.get()
        if features_str == "kill":
            LOGGER.info('write_process-{} finished'.format(os.getpid()))
            break

        if batch_count >= w_batch_num:
            wf.flush()
            wf.close()
            file_count += 1
            wf = open("/".join([write_dir, str(file_count) + ".tsv"]), "w")
            batch_count = 0
        for one_features_str in features_str:
            wf.write(one_features_str + "\n")
        batch_count += 1


def _write_featurestr(write_fp, featurestr_q, w_batch_num=10000, is_dir=False):
    if is_dir:
        _write_featurestr_to_dir(write_fp, featurestr_q, w_batch_num)
    else:
        _write_featurestr_to_file(write_fp, featurestr_q)
# =======================================================================================


def _fill_files_queue(fast5s_q, fast5_files, batch_size, is_single=False):
    batch_size_tmp = 1 if not is_single else batch_size
    for i in np.arange(0, len(fast5_files), batch_size_tmp):
        fast5s_q.put(fast5_files[i:(i+batch_size_tmp)])
    return


def _read_position_file(position_file):
    postions = set()
    with open(position_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            postions.add(key_sep.join(words[:3]))
    return postions


def check_basecallgroup(fast5_fn, basecall_group, basecall_subgroup, is_single=False):
    if is_single:
        fast5read = fast5_reader.SingleFast5(fast5_fn, is_single)
    else:
        mfast5reads = fast5_reader.MultiFast5(fast5_fn)
        readname = next(iter(mfast5reads))
        fast5read = fast5_reader.SingleFast5(mfast5reads[readname],
                                             is_single,
                                             readname)
    bgrp = basecall_group if basecall_group is not None else fast5read.get_lastest_basecallgroup()

    seggrp = fast5read.get_basecallgroup_related_sementation(bgrp)
    fast5read.close()
    if not is_single:
        mfast5reads.close()
    LOGGER.info("basecall_group: {}/{}, segmentation_group: {}".format(bgrp, basecall_subgroup, seggrp))


def _extract_preprocess(fast5_dir, is_recursive, motifs, is_dna, reference_path, f5_batch_num,
                        position_file, args):

    fast5_files = get_fast5s(fast5_dir, is_recursive)
    if args.single:
        LOGGER.info("{} fast5s/reads in total".format(len(fast5_files)))
    else:
        LOGGER.info("{} multi-fast5s in total".format(len(fast5_files)))
    check_basecallgroup(fast5_files[0], args.basecall_group, args.basecall_subgroup, args.single)

    LOGGER.info("parse the motifs string")
    motif_seqs = get_motif_seqs(motifs, is_dna)

    LOGGER.info("read genome reference file")
    if args.mapping:
        chrom2len = get_contig2len(reference_path)
        contigs = None
    else:
        chrom2len, contigs = get_contig2len_n_seq(reference_path)

    LOGGER.info("read position file if it is not None")
    positions = None
    if position_file is not None:
        positions = _read_position_file(position_file)

    fast5s_q = Queue()
    _fill_files_queue(fast5s_q, fast5_files, f5_batch_num, args.single)

    return motif_seqs, chrom2len, fast5s_q, len(fast5_files), positions, contigs


# for call_mods module
def _batchlize_features_list(features_list):
    sampleinfo = []  # contains: chromosome, pos, strand, pos_in_strand, read_name, read_loc
    kmers = []
    base_means = []
    base_stds = []
    base_signal_lens = []
    base_probs = []
    k_signals = []
    labels = []
    for features in features_list:
        chrom, pos, alignstrand, loc_in_ref, readname, read_loc, k_mer, signal_means, signal_stds, \
            signal_lens, kmer_probs, kmer_base_signals, f_methy_label = features

        sampleinfo.append("\t".join([chrom, str(pos), alignstrand, str(loc_in_ref), readname,
                                     str(read_loc)]))
        kmers.append([base2code_dna[x] for x in k_mer])
        base_means.append(signal_means)
        base_stds.append(signal_stds)
        base_signal_lens.append(signal_lens)
        base_probs.append(kmer_probs)
        k_signals.append(kmer_base_signals)
        labels.append(f_methy_label)
    features_batches = (sampleinfo, kmers, base_means, base_stds,
                        base_signal_lens, base_probs, k_signals, labels)
    return features_batches


# pipe process, minimap2 ==========================================================================
def _process_read_map(fast5_path, extract_conn,
                  basecall_group, basecall_subgroup, norm_method,
                  mapq, identity, coverage_ratio,
                  motif_seqs, chrom2len, positions, read_strand,
                  mod_loc, seq_len, signal_len, methy_label,
                  pad_only_r,
                  is_single, readname,is_trace):
    # input: f5_path
    # output: features_list
    # -1: read fast5 error, -2: align read error, -3: quality too low
    success, readinfo = _get_read_signalinfo(fast5_path, basecall_group, basecall_subgroup, norm_method,
                                             is_single, readname,
                                             is_mapping=True, corrected_group="",is_trace=is_trace)
    if success == 0:
        return -1, None

    # base_probs: array of (read_len, 4)
    read_id, baseseq, signal_group, base_probs, _ = readinfo
    extract_conn.send((read_id, baseseq))
    success, map_res = extract_conn.recv()
    if success == 0:
        return -2, None

    map_res = MAP_RES(*map_res)
    if map_res.mapq < mapq:
        LOGGER.debug("mapq too low: {}, mapq: {}".format(map_res.read_id, map_res.mapq))
        return -3, None
    if _compute_pct_identity(map_res.cigar) < identity:
        LOGGER.debug("identity too low: {}, identity: {}".format(map_res.read_id,
                                                                 _compute_pct_identity(map_res.cigar)))
        return -3, None
    if (map_res.q_en - map_res.q_st) / float(len(map_res.q_seq)) < coverage_ratio:
        return -3, None
    try:
        r_to_q_poss = parse_cigar(map_res.cigar, map_res.strand, len(map_res.ref_seq))
    except Exception:
        LOGGER.debug("cigar parsing error: {}".format(map_res.read_id))
        return 0, None

    ref_signal_grp = [None, ] * len(map_res.ref_seq)
    ref_baseprobs = [0., ] * len(map_res.ref_seq)
    ref_readlocs = [0, ] * len(map_res.ref_seq)
    for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
        # signal groups
        ref_signal_grp[ref_pos] = signal_group[q_pos + map_res.q_st]
        ref_readlocs[ref_pos] = q_pos + map_res.q_st
        # trace
        try:
            base_idx = base2code_dna[map_res.ref_seq[ref_pos]]
            ref_baseprobs[ref_pos] = base_probs[q_pos + map_res.q_st][base_idx]
        except Exception:
            #LOGGER.debug("error when extracting trace feature: {}-{}".format(map_res.read_id,
            #                                                                 q_pos))
            ref_baseprobs[ref_pos] = []
            continue

    # ref_mapinfo = (read_id, map_res.ref_seq, map_res.ctg, map_res.strand, map_res.r_st, map_res.r_en,
    #                ref_signal_grp, ref_baseprobs)
    ref_mapinfo = (read_id, map_res.ref_seq, ref_readlocs, ref_signal_grp, ref_baseprobs,
                   [(len(map_res.ref_seq), map_res.ctg, map_res.strand, map_res.r_st, map_res.r_en), ])
    features_list = _extract_features(ref_mapinfo, motif_seqs, chrom2len, positions, read_strand,
                                      mod_loc, seq_len, signal_len, methy_label,
                                      pad_only_r)
    return 1, features_list


def _process_read_nomap(fast5_path, basecall_group, basecall_subgroup, norm_method,
                        mapq, identity, coverage_ratio,
                        motif_seqs, chrom2len, positions,
                        read_strand, mod_loc, seq_len, signal_len, methy_label,
                        pad_only_r,
                        is_single, readname,
                        corrected_group, chroms,is_trace):
    # input: f5_path
    # output: features_list
    # -1: read fast5 error, -2: align read error, -3: quality too low
    success, readinfo = _get_read_signalinfo(fast5_path, basecall_group, basecall_subgroup, norm_method,
                                             is_single, readname,
                                             is_mapping=False, corrected_group=corrected_group,is_trace=is_trace)
    if success == 0:
        return -1, None

    # base_probs: array of (read_len, 4)
    read_id, baseseq, signal_group, base_probs, mapinfo = readinfo
    if len(mapinfo) == 0:
        return -2, None
    failed_cnt = 0

    combed_ref_seq = ""
    combed_ref_signal_grp = []
    combed_ref_baseprobs = []
    combed_ref_readlocs = []
    combed_cigartuples = []
    seg_mapping = []
    for map_idx in range(len(mapinfo)):
        item_cigar, item_chrom, item_strand, item_read_s, item_read_e, item_ref_s, item_ref_e = mapinfo[map_idx]
        cigartuple = _convert_cigarstring2tuple(item_cigar)
        # WARN: remove D(eltion, encoded as 2) at the end of the alignment?
        if map_idx == len(mapinfo) - 1:
            if item_strand == "+" and cigartuple[-1][-1] == 2:
                oplen_tmp = cigartuple[-1][0]
                cigartuple = cigartuple[:-1]
                item_ref_e -= oplen_tmp
            if item_strand == "-" and cigartuple[0][-1] == 2:
                oplen_tmp = cigartuple[0][0]
                cigartuple = cigartuple[1:]
                item_ref_s += oplen_tmp
        try:
            if item_chrom == 'read':
                assert item_strand == "+"
                ref_seq = baseseq[item_ref_s:item_ref_e]
                strand_code = 1
                r_to_q_poss = parse_cigar(cigartuple, strand_code, len(ref_seq))
            else:
                ref_seq = chroms[item_chrom][item_ref_s:item_ref_e]
                if item_strand == "-":
                    ref_seq = complement_seq(ref_seq)
                strand_code = 0 if item_strand == "-" else 1
                r_to_q_poss = parse_cigar(cigartuple, strand_code, len(ref_seq))
        except KeyError:
            LOGGER.debug("no chrom-{} in reference genome: {}".format(item_chrom, read_id))
            failed_cnt += 1
            continue
        except ValueError:
            LOGGER.debug("cigar parsing error: {}".format(read_id))
            failed_cnt += 1
            continue

        ref_signal_grp = [None, ] * len(ref_seq)
        ref_baseprobs = [0., ] * len(ref_seq)
        ref_readlocs = [0, ] * len(ref_seq)
        for ref_pos, q_pos in enumerate(r_to_q_poss[:-1]):
            # signal groups
            ref_readlocs[ref_pos] = q_pos + item_read_s
            ref_signal_grp[ref_pos] = signal_group[q_pos + item_read_s]
            # trace
            try:
                base_idx = base2code_dna[ref_seq[ref_pos]]
                ref_baseprobs[ref_pos] = base_probs[q_pos + item_read_s][base_idx]
            except Exception:
                #LOGGER.debug("error when extracting trace feature: {}-{}".format(read_id,
                #                                                                 q_pos))
                ref_baseprobs[ref_pos] = []
                continue

        combed_ref_seq += ref_seq
        combed_ref_readlocs += ref_readlocs
        combed_ref_signal_grp += ref_signal_grp
        combed_ref_baseprobs += ref_baseprobs
        combed_cigartuples += cigartuple
        seg_mapping.append((len(ref_seq), item_chrom, strand_code, item_ref_s, item_ref_e))
    if failed_cnt > 0:
        return -3, None
    try:
        if _compute_pct_identity(combed_cigartuples) < identity:
            LOGGER.debug("identity too low: {}, identity: {}".format(read_id,
                                                                     _compute_pct_identity(combed_cigartuples)))
            return -3, None
    except ZeroDivisionError:
        raise ZeroDivisionError("{}, {}".format(read_id, combed_cigartuples))
    del combed_cigartuples
    ref_mapinfo = (read_id, combed_ref_seq, combed_ref_readlocs,
                   combed_ref_signal_grp, combed_ref_baseprobs, seg_mapping)
    features_list = _extract_features(ref_mapinfo, motif_seqs, chrom2len, positions, read_strand,
                                      mod_loc, seq_len, signal_len, methy_label,
                                      pad_only_r)
    return 1, features_list


def _process_read(fast5_path, extract_conn,
                  basecall_group, basecall_subgroup, norm_method,
                  mapq, identity, coverage_ratio,
                  motif_seqs, chrom2len, positions, read_strand,
                  mod_loc, seq_len, signal_len, methy_label,
                  pad_only_r,
                  is_single, readname,
                  is_mapping, corrected_group, chroms,is_trace):
    if is_mapping:
        return _process_read_map(fast5_path, extract_conn,
                                 basecall_group, basecall_subgroup, norm_method,
                                 mapq, identity, coverage_ratio,
                                 motif_seqs, chrom2len, positions, read_strand,
                                 mod_loc, seq_len, signal_len, methy_label,
                                 pad_only_r,
                                 is_single, readname,is_trace)
    else:
        return _process_read_nomap(fast5_path, basecall_group, basecall_subgroup, norm_method,
                                   mapq, identity, coverage_ratio,
                                   motif_seqs, chrom2len, positions, read_strand,
                                   mod_loc, seq_len, signal_len, methy_label,
                                   pad_only_r,
                                   is_single, readname,
                                   corrected_group, chroms,is_trace)


def _process_reads(fast5s_q, features_q, error_q, extract_conn,
                   basecall_group, basecall_subgroup, norm_method,
                   mapq, identity, coverage_ratio,
                   motif_seqs, chrom2len, positions, read_strand,
                   mod_loc, seq_len, signal_len, methy_label,
                   pad_only_r,
                   is_single, f5_batch_num,
                   is_mapping, corrected_group, chroms,
                   is_to_str=True, is_batchlize=False,is_trace=False):
    assert not (is_to_str and is_batchlize)
    LOGGER.info("extract_features process-{} starts".format(os.getpid()))
    f5_num = 0  # number of reads
    error2num = {-1: 0, -2: 0, -3: 0, 0: 0}  # (-1, -2, -3, 0)
    cnt_fast5batch = 0
    while True:
        if fast5s_q.empty():
            time.sleep(time_wait)
        fast5s = fast5s_q.get()
        if fast5s == "kill":
            fast5s_q.put("kill")
            break
        # single or multi-format fast5
        if is_single:
            f5_num += len(fast5s)
            features_list_batch = []
            for f5_path in fast5s:
                success, features_list = _process_read(f5_path, extract_conn,
                                                       basecall_group, basecall_subgroup, norm_method,
                                                       mapq, identity, coverage_ratio,
                                                       motif_seqs, chrom2len, positions, read_strand,
                                                       mod_loc, seq_len, signal_len, methy_label,
                                                       pad_only_r, is_single, "",
                                                       is_mapping, corrected_group, chroms,is_trace)
                if success <= 0:
                    error2num[success] += 1
                else:
                    if is_to_str:
                        features_list_batch += [_features_to_str(features) for features in features_list]
                    else:
                        features_list_batch += features_list
            if not is_to_str and is_batchlize:  # if is_to_str, then ignore is_batchlize
                features_list_batch = _batchlize_features_list(features_list_batch)
            features_q.put(features_list_batch)
            while features_q.qsize() > queue_size_border:
                time.sleep(time_wait)
        else:
            features_list_batch = []
            for f5_path in fast5s:
                multi_f5 = fast5_reader.MultiFast5(f5_path)
                read_cnt = 0
                for readname in iter(multi_f5):
                    read_cnt += 1
                    singlef5 = multi_f5[readname]
                    success, features_list = _process_read(singlef5, extract_conn,
                                                           basecall_group, basecall_subgroup, norm_method,
                                                           mapq, identity, coverage_ratio,
                                                           motif_seqs, chrom2len, positions, read_strand,
                                                           mod_loc, seq_len, signal_len, methy_label,
                                                           pad_only_r, is_single, readname,
                                                           is_mapping, corrected_group, chroms,is_trace)
                    if success <= 0:
                        error2num[success] += 1
                    else:
                        if is_to_str:
                            features_list_batch += [_features_to_str(features) for features in features_list]
                        else:
                            features_list_batch += features_list

                    if read_cnt % f5_batch_num == 0 and len(features_list_batch) > 0:
                        if not is_to_str and is_batchlize:  # if is_to_str, then ignore is_batchlize
                            features_list_batch = _batchlize_features_list(features_list_batch)
                        features_q.put(features_list_batch)
                        while features_q.qsize() > queue_size_border:
                            time.sleep(time_wait)
                        features_list_batch = []
                if len(features_list_batch) > 0:
                    if not is_to_str and is_batchlize:  # if is_to_str, then ignore is_batchlize
                        features_list_batch = _batchlize_features_list(features_list_batch)
                    features_q.put(features_list_batch)
                    while features_q.qsize() > queue_size_border:
                        time.sleep(time_wait)
                    features_list_batch = []
                multi_f5.close()
                f5_num += read_cnt
        cnt_fast5batch += 1
        if cnt_fast5batch % 100 == 0:
            LOGGER.info("extrac_features process-{}, {} fast5_batches "
                        "proceed".format(os.getpid(), cnt_fast5batch))
    error_q.put(error2num)
    error_total = sum([error2num[error_code] for error_code in error2num.keys()])
    LOGGER.info("extract_features process-{} finished, proceed {} reads, failed: {}".format(os.getpid(),
                                                                                            f5_num,
                                                                                            error_total))


def start_extract_processes(fast5s_q, features_q, error_q, nproc,
                            basecall_group, basecall_subgroup, norm_method,
                            mapq, identity, coverage_ratio,
                            motif_seqs, chrom2len, positions, read_strand,
                            mod_loc, seq_len, signal_len, methy_label,
                            pad_only_r, is_single, f5_batch_num,
                            is_mapping, corrected_group, chroms,
                            is_to_str=True, is_batchlize=False,is_trace=False):
    random.seed(1234)
    extract_ps, map_conns = [], []
    for proc_idx in range(nproc):
        if is_mapping:
            map_conn, extract_conn = mp.Pipe()
            map_conns.append(map_conn)
        else:
            extract_conn = None
        p = mp.Process(target=_process_reads,
                       args=(fast5s_q, features_q, error_q, extract_conn,
                             basecall_group, basecall_subgroup, norm_method,
                             mapq, identity, coverage_ratio,
                             motif_seqs, chrom2len, positions, read_strand,
                             mod_loc, seq_len, signal_len, methy_label,
                             pad_only_r,
                             is_single, f5_batch_num,
                             is_mapping, corrected_group, chroms,
                             is_to_str, is_batchlize,is_trace),
                       name="extracter_{:03d}".format(proc_idx))
        p.daemon = True
        p.start()
        extract_ps.append(p)

        if extract_conn is not None:
            extract_conn.close()
        del extract_conn
    return extract_ps, map_conns


def start_map_threads(map_conns, aligner):
    time.sleep(1)
    map_read_ts = []
    for ti, map_conn in enumerate(map_conns):
        map_read_ts.append(threading.Thread(
            target=_map_read_to_ref_process, args=(aligner, map_conn),
            daemon=True, name='aligner_{:03d}'.format(ti)))
        map_read_ts[-1].start()
    return map_read_ts
# =================================================================================


def _reads_processed_stats(error2num, len_fast5s, is_single=False):
    error_total = sum([error2num[error_code] for error_code in error2num.keys()])
    if len_fast5s == 0:
        LOGGER.error("no fast5 file fonud in --fast5_dir")
        return
    if is_single:
        LOGGER.info("summary:\n"
                    "  total fast5s: {}\n"
                    "  failed fast5s: {}({:.1f}%)\n"
                    "    error in reading fast5s: {}({:.1f}%)\n"
                    "    error in alignment: {}({:.1f}%)\n"
                    "    low quality: {}({:.1f}%)\n"
                    "    error in parsing cigar: {}({:.1f}%)\n".format(len_fast5s,
                                                                       error_total,
                                                                       error_total/float(len_fast5s) * 100,
                                                                       error2num[-1],
                                                                       error2num[-1]/float(len_fast5s) * 100,
                                                                       error2num[-2],
                                                                       error2num[-2]/float(len_fast5s) * 100,
                                                                       error2num[-3],
                                                                       error2num[-3]/float(len_fast5s) * 100,
                                                                       error2num[0],
                                                                       error2num[0]/float(len_fast5s) * 100))
    else:
        LOGGER.info("summary:\n"
                    "  total fast5s: {}\n"
                    "  failed reads: {}\n".format(len_fast5s,
                                                  error_total))


def extract_features(args):
    start = time.time()
    ref_path = os.path.abspath(args.reference_path)
    if not os.path.exists(ref_path):
        raise ValueError("--reference_path not set right!")

    LOGGER.info("[extract] starts")

    fast5_dir = os.path.abspath(args.fast5_dir)
    if not os.path.exists(fast5_dir):
        raise ValueError("--fast5_dir not set right!")
    if not os.path.isdir(fast5_dir):
        raise NotADirectoryError("--fast5_dir not a directory")
    is_recursive = str2bool(args.recursively)
    is_dna = False if args.rna else True
    motif_seqs, chrom2len, fast5s_q, len_fast5s, \
        positions, contigs = _extract_preprocess(fast5_dir, is_recursive,
                                                 args.motifs, is_dna, ref_path,
                                                 args.f5_batch_size, args.positions,
                                                 args)

    # read_strand has been deprecated
    read_strand = _get_read_sequened_strand(args.basecall_subgroup)

    fast5s_q.put("kill")
    features_q = Queue()
    error_q = Queue()

    if args.mapping:
        aligner = get_aligner(ref_path, args.best_n)

    nproc = args.nproc
    if nproc < 2:
        nproc = 2
    nproc_extr = nproc - 1

    extract_ps, map_conns = start_extract_processes(fast5s_q, features_q, error_q, nproc_extr,
                                                    args.basecall_group, args.basecall_subgroup,
                                                    args.normalize_method,
                                                    args.mapq, args.identity, args.coverage_ratio,
                                                    motif_seqs, chrom2len, positions, read_strand,
                                                    args.mod_loc, args.seq_len, args.signal_len, args.methy_label,
                                                    args.pad_only_r, args.single, args.f5_batch_size,
                                                    args.mapping, args.corrected_group, contigs,True,False,args.trace)

    p_w = mp.Process(target=_write_featurestr, args=(args.write_path, features_q, args.w_batch_num,
                                                     str2bool(args.w_is_dir)),
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
    features_q.put("kill")

    p_w.join()

    _reads_processed_stats(error2num, len_fast5s, args.single)
    LOGGER.info("[extract] finished, cost {:.1f}s".format(time.time()-start))


def main():
    extraction_parser = argparse.ArgumentParser("extract features from guppy FAST5s for "
                                                "training or testing."
                                                "\nIt is suggested that running this module 1 flowcell a time, "
                                                "or a group of flowcells a time, "
                                                "if the whole data is extremely large.")
    ep_input = extraction_parser.add_argument_group("INPUT")
    ep_input.add_argument("--fast5_dir", "-i", action="store", type=str,
                          required=True,
                          help="the directory of fast5 files")
    ep_input.add_argument("--recursively", "-r", action="store", type=str, required=False,
                          default='yes',
                          help='is to find fast5 files from fast5_dir recursively. '
                               'default true, t, yes, 1')
    ep_input.add_argument("--single", action="store_true", default=False, required=False,
                          help='the fast5 files are in single-read format')
    ep_input.add_argument("--reference_path", action="store",
                          type=str, required=True,
                          help="the reference file to be used, usually is a .fa file")
    ep_input.add_argument("--rna", action="store_true", default=False, required=False,
                          help='the fast5 files are from RNA samples. if is rna, the signals are reversed. '
                               'NOTE: Currently no use, waiting for further extentsion')
    
    ep_extraction = extraction_parser.add_argument_group("EXTRACTION")
    ep_extraction.add_argument("--basecall_group", action="store", type=str, required=False,
                               default=None,
                               help='basecall group generated by Guppy. e.g., Basecall_1D_000')
    ep_extraction.add_argument("--basecall_subgroup", action="store", type=str, required=False,
                               default='BaseCalled_template',
                               help='the basecall subgroup of fast5 files. default BaseCalled_template')
    ep_extraction.add_argument("--normalize_method", action="store", type=str, choices=["mad", "zscore"],
                               default="mad", required=False,
                               help="the way for normalizing signals in read level. "
                                    "mad or zscore, default mad")
    ep_extraction.add_argument("--methy_label", action="store", type=int,
                               choices=[1, 0], required=False, default=1,
                               help="the label of the interested modified bases, this is for training."
                                    " 0 or 1, default 1")
    ep_extraction.add_argument("--seq_len", action="store",
                               type=int, required=False, default=13,
                               help="len of kmer. default 13")
    ep_extraction.add_argument("--signal_len", action="store",
                               type=int, required=False, default=15,
                               help="the number of signals of one base to be used in deepsignal_plant, default 15")
    ep_extraction.add_argument("--motifs", action="store", type=str,
                               required=False, default='CG',
                               help='motif seq to be extracted, default: CG. '
                                    'can be multi motifs splited by comma '
                                    '(no space allowed in the input str), '
                                    'or use IUPAC alphabet, '
                                    'the mod_loc of all motifs must be '
                                    'the same')
    ep_extraction.add_argument("--trace", action="store_true", default=False, required=False,
                            help='use trace, default false')
    ep_extraction.add_argument("--mod_loc", action="store", type=int, required=False, default=0,
                               help='0-based location of the targeted base in the motif, default 0')
    # ep_extraction.add_argument("--region", action="store", type=str,
    #                            required=False, default=None,
    #                            help="region of interest, e.g.: chr1:0-10000, default None, "
    #                                 "for the whole region")
    ep_extraction.add_argument("--positions", action="store", type=str,
                               required=False, default=None,
                               help="file with a list of positions interested (must be formatted as tab-separated file"
                                    " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
                                    "need to be set. --positions is used to narrow down the range of the trageted "
                                    "motif locs. default None")
    ep_extraction.add_argument("--f5_batch_size", action="store", type=int, default=50,
                               required=False,
                               help="number of files to be processed by each process one time, default 50")
    ep_extraction.add_argument("--pad_only_r", action="store_true", default=False,
                               help="pad zeros to only the right of signals array of one base, "
                               "when the number of signals is less than --signal_len. "
                               "default False (pad in two sides).")

    ep_mape = extraction_parser.add_argument_group("MAPe")
    ep_mape.add_argument("--corrected_group", action="store", type=str, required=False,
                         default='RawGenomeCorrected_000',
                         help='the corrected_group of fast5 files, '
                              'default RawGenomeCorrected_000')

    ep_mapping = extraction_parser.add_argument_group("MAPPING")
    ep_mapping.add_argument("--mapping", action="store_true", default=False, required=False,
                            help='use MAPPING to get alignment, default false')
    ep_mapping.add_argument("--mapq", type=int, default=10, required=False,
                            help="MAPping Quality cutoff for selecting alignment items, default 10")
    ep_mapping.add_argument("--identity", type=float, default=0.75, required=False,
                            help="identity cutoff for selecting alignment items, default 0.75")
    ep_mapping.add_argument("--coverage_ratio", type=float, default=0.75, required=False,
                            help="percent of coverage, read alignment len against read len, default 0.75")
    ep_mapping.add_argument("--best_n", "-n", type=int, default=1, required=False,
                            help="best_n arg in mappy(minimap2), default 1")

    ep_output = extraction_parser.add_argument_group("OUTPUT")
    ep_output.add_argument("--write_path", "-o", action="store",
                           type=str, required=True,
                           help='file path to save the features')
    ep_output.add_argument("--w_is_dir", action="store",
                           type=str, required=False, default="no",
                           help='if using a dir to save features into multiple files')
    ep_output.add_argument("--w_batch_num", action="store",
                           type=int, required=False, default=200,
                           help='features batch num to save in a single writed file when --is_dir is true')

    extraction_parser.add_argument("--nproc", "-p", action="store", type=int, default=10,
                                   required=False,
                                   help="number of processes to be used, default 10")

    extraction_args = extraction_parser.parse_args()
    display_args(extraction_args)

    extract_features(extraction_args)


if __name__ == '__main__':
    sys.exit(main())
