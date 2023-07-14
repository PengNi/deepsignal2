import multiprocessing as mp
import numpy as np
import pod5 as p5
import pysam
import sys
from multiprocessing import Queue  # ,Manager,Pool
import time
import os
import signal
import logging
from tqdm import tqdm
from multiprocessing import Pool
import random
from statsmodels import robust
from utils.utils import parse_args
from utils.log import get_logger, init_logger
from utils import constants

args = parse_args()
signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
logger = logging.getLogger("test_logger")
logger.setLevel(logging.DEBUG)
test_log = logging.FileHandler(
    args.log_file, "a", encoding="utf-8"
)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s"
)
test_log.setFormatter(formatter)
# 加载文件到logger对象中
logger.addHandler(test_log)


def extract_signal_from_pod5(pod5_path) -> np.array:
    signals = []
    with p5.Reader(pod5_path) as reader:
        for read_record in reader.reads():
            if read_record.signal is None:
                logger.critical(
                    "Signal is None for read id {}".format(read_record.read_id)
                )
            # signals[str(read_record.read_id)] =
            # {'signal':read_record.signal,'shift':read_record.calibration.offset,'scale':read_record.calibration.scale}#不加str会变成UUID，很奇怪
            signals.append(
                [
                    str(read_record.read_id),
                    read_record.signal.astype(np.int16),
                    # np.int16(read_record.calibration.offset),
                    # np.float16(read_record.calibration.scale),
                ]
            )
            # 0:read_id,1:signal,2:shift,3:scale
    return np.array(signals, dtype=object)  # np.array is small than list


def get_read_ids(bam_idx, pod5_fh, num_reads):
    """Get overlapping read ids from bam index and pod5 file

    Args:
        bam_idx (ReadIndexedBam): Read indexed BAM
        pod5_fh (pod5.Reader): POD5 file handle
        num_reads (int): Maximum number of reads, or None for no max
    """
    logger.info("Extracting read IDs from POD5")
    pod5_read_ids = set(pod5_fh.read_ids)
    num_pod5_reads = len(pod5_read_ids)
    # pod5 will raise when it cannot find a "selected" read id, so we make
    # sure they're all present before starting
    # todo(arand) this could be performed using the read_table instead, but
    #  it's worth checking that it's actually faster and doesn't explode
    #  memory before switching from a sweep throug the pod5 file
    both_read_ids = list(pod5_read_ids.intersection(bam_idx.read_ids))
    num_both_read_ids = len(both_read_ids)
    logger.info(
        f"Found {bam_idx.num_reads} BAM records, {num_pod5_reads} "
        f"POD5 reads, and {num_both_read_ids} in common"
    )
    if num_reads is None:
        num_reads = num_both_read_ids
    else:
        num_reads = min(num_reads, num_both_read_ids)
    return both_read_ids, num_reads


def extract_move_from_bam(bam_path) -> np.array:
    seq_move = []
    bamfile = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    try:
        for read in bamfile.fetch(until_eof=True):  # 暂时不使用索引，使用返回是空值
            # print(read.query_name)
            tags = dict(read.tags)
            mv_tag = tags["mv"]
            ts_tag = tags["ts"]
            sm_tag = tags["sm"]
            sd_tag = tags["sd"]
            # read.update({read.query_name:{"sequence":read.query_sequence,"stride":mv_tag[0],"mv_table":np.array(mv_tag[1:]),"num_trimmed":ts_tag,"shift":sm_tag,"scale":sd_tag}})
            seq_move.append(
                [
                    read.query_name,
                    read.query_sequence,
                    np.int16(mv_tag[0]),
                    np.array(mv_tag[1:], dtype=np.int16),
                    np.int16(ts_tag),
                    read.reference_name,
                    np.int64(read.reference_start),
                    "-" if read.is_reverse else "+",
                    # np.float16(sm_tag),
                    # np.float16(sd_tag),
                ]
            )
    except ValueError:
        print("bam don't has index")
        for read in bamfile.fetch(until_eof=True, multiple_iterators=False):
            tags = dict(read.tags)
            mv_tag = tags["mv"]
            ts_tag = tags["ts"]
            sm_tag = tags["sm"]
            sd_tag = tags["sd"]
            seq_move.append(
                [
                    read.query_name,
                    read.query_sequence,
                    np.int16(mv_tag[0]),
                    np.array(mv_tag[1:], dtype=np.int16),
                    np.int16(ts_tag),
                    read.reference_name,
                    np.int64(read.reference_start),
                    "-" if read.is_reverse else "+",
                    # np.float16(sm_tag),
                    # np.float16(sd_tag),
                ]
            )
            # 0:read_id,1:sequence,2:stride,3:mv_table,4:num_trimmed,5:to_norm_shift,6:to_norm_scale
            # read[read.query_name] = {"sequence":read.query_sequence,"stride":mv_tag[0],"mv_table":np.array(mv_tag[1:]),"num_trimmed":ts_tag,"shift":sm_tag,"scale":sd_tag}
    return np.array(seq_move, dtype=object)


def read_from_pod5_bam(pod5_path, bam_path, read_id=None) -> np.array:
    read = []
    signal = extract_signal_from_pod5(pod5_path)
    seq_move = extract_move_from_bam(bam_path)
    if read_id is not None:
        for i in range(len(seq_move)):
            if seq_move[i][0] == read_id:
                if seq_move[i][1] is not None:
                    for j in range(len(signal)):
                        if signal[j][0] == seq_move[i][0]:
                            read.append(
                                [
                                    signal[j][constants.POD5_READ_ID],
                                    signal[j][constants.POD5_SIGNAL],
                                    # signal[j][constants.POD5_SHIFT],
                                    # signal[j][constants.POD5_SCALE],
                                    seq_move[i][constants.BAM_SEQ],
                                    seq_move[i][constants.BAM_STRIDE],
                                    seq_move[i][constants.BAM_MV_TABLE],
                                    seq_move[i][constants.BAM_NUM_TRIMMED],
                                    seq_move[i][constants.BAM_TO_NORM_SHIFT],
                                    seq_move[i][constants.BAM_TO_NORM_SCALE],
                                    seq_move[i][constants.BAM_REF_NAME],
                                    seq_move[i][constants.BAM_REF_START],
                                    seq_move[i][constants.BAM_REF_STRAND],
                                ]
                            )

    else:
        for i in range(len(seq_move)):
            if seq_move[i][1] is not None:
                for j in range(len(signal)):
                    if signal[j][0] == seq_move[i][0]:
                        read.append(
                            [
                                signal[j][constants.POD5_READ_ID],
                                signal[j][constants.POD5_SIGNAL],
                                # signal[j][constants.POD5_SHIFT],
                                # signal[j][constants.POD5_SCALE],
                                seq_move[i][constants.BAM_SEQ],
                                seq_move[i][constants.BAM_STRIDE],
                                seq_move[i][constants.BAM_MV_TABLE],
                                seq_move[i][constants.BAM_NUM_TRIMMED],
                                seq_move[i][constants.BAM_TO_NORM_SHIFT],
                                seq_move[i][constants.BAM_TO_NORM_SCALE],
                                seq_move[i][constants.BAM_REF_NAME],
                                seq_move[i][constants.BAM_REF_START],
                                seq_move[i][constants.BAM_REF_STRAND],
                            ]
                        )
                # 0:read_id,1:signal,2:to_pA_shift,3:to_pA_scale,4:sequence,5:stride,6:mv_table,7:num_trimmed,8:to_norm_shift,9:to_norm_scale

    return np.array(read, dtype=object)


def get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif=0) -> list:
    """

    :param seqstr:
    :param motifset:
    :param methyloc_in_motif: 0-based
    :return:
    """
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i: i + motiflen] in motifset:
            sites.append(i + methyloc_in_motif)
    return sites


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(constants.iupac_alphabets[bbase])
        else:
            outbases.append(constants.iupac_alphabets_rna[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        elif len(bases_list) == 2:
            pseqs = []
            for fbase in bases_list[0]:
                for sbase in bases_list[1]:
                    pseqs.append(fbase + sbase)
            return pseqs
        else:
            pseqs = recursive_permute(bases_list[1:])
            pseq_list = [bases_list[0], pseqs]
            return recursive_permute(pseq_list)

    return recursive_permute(outbases)


def get_motif_seqs(motifs, is_dna=True):
    ori_motif_seqs = motifs.strip().split(",")

    motif_seqs = []
    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif.strip().upper(), is_dna)
    return motif_seqs


def expand(feature, index, nbase, nsig, num, fill_num=1):
    # nbase.append(np.tile(np.array(feature[index][5],dtype=str),num*fill_num))
    for i in range(fill_num):
        t = np.tile(np.array(feature[index][5], dtype=str), num)
        nbase.append(t)
    # nstd.append(np.tile(feature[index][2],num*fill_num))
    # nmean.append(np.tile(feature[index][3],num*fill_num))
    try:
        # nsig.append(np.tile(np.random.choice(feature[index][1],size=num,replace=False),fill_num))
        for i in range(fill_num):
            t = np.random.choice(feature[index][1], size=num, replace=False)
            nsig.append(t)
        # np.array取随机不能用random包，要用numpy自带的random
    except Exception as e:
        logger.critical(feature[index][1])
    # return nbase,nsig


# 0:read_id,1:signal,2:std,3:mean,4:num,5:base
def _get_neighbord_feature(sequence, feature, base_num) -> list:
    # 数据预处理主要速度瓶颈，同样的reads数，不运行这个函数大概快了十倍，从二十多分钟减到两分钟
    motif = "CG"
    # max_sites=15
    motif_seqs = get_motif_seqs(motif)
    tsite_locs = get_refloc_of_methysite_in_motif(sequence, set(motif_seqs))
    # if len(tsite_locs)>max_sites:
    #    tsite_locs = np.random.choice(
    #            tsite_locs,
    #            size=max_sites,
    #            replace=False,
    #        )

    nfeature = []
    windows_size = (base_num - 1) // 2
    signal_sample = 5
    for i in range(len(feature)):
        nbase = []
        # nstd=[]
        # nmean=[]
        nsig = []
        if i not in tsite_locs:
            continue
        # 更改扩增逻辑，增添采样函数
        # remora一条read好像只提取15个点
        # if feature[i][4]>base_num:
        #    logger.info("base correspoding signal number {} is more than window size {}".format(feature[i][4],base_num))

        if i < windows_size:
            flag = windows_size - i
            expand(feature, i, nbase, nsig, signal_sample, windows_size - i)
            if i != 0:
                for k in range(i):  # 左闭右开
                    expand(feature, k, nbase, nsig, signal_sample)
                    flag += 1
            for k in range(i, i + windows_size + 1):
                expand(feature, k, nbase, nsig, signal_sample)
                flag += 1
            logger.debug(
                "focus base on the far left of read and expand number is {}".format(
                    flag
                )
            )
        elif (len(feature) - 1) - i < windows_size:
            flag = 0
            for k in range(i - windows_size, i + 1):
                flag += 1
                expand(feature, k, nbase, nsig, signal_sample)
            if i != len(feature) - 1:
                for k in range(i + 1, len(feature)):
                    flag += 1
                    expand(feature, k, nbase, nsig, signal_sample)
            flag += windows_size - ((len(feature) - 1) - i)
            expand(
                feature,
                i,
                nbase,
                nsig,
                signal_sample,
                windows_size - ((len(feature) - 1) - i),
            )
            logger.debug(
                "focus base on the far right of read and expand number is {}".format(
                    flag
                )
            )
        else:
            flag = 0
            for k in range(i - windows_size, i):
                expand(feature, k, nbase, nsig, signal_sample)
                flag += 1
            for k in range(i, i + windows_size + 1):
                expand(feature, k, nbase, nsig, signal_sample)
                flag += 1
            if flag != base_num:
                logger.error("focus base expand number is {}".format(flag))
        # feature[read_id][i].update({'nbase':nbase,'nsig':nsig,'nstd':nstd,'nmean':nmean})
        nfeature.append([feature[i][0], nbase, nsig])

        # 0:read_id,1:nbase,2:nsig,3:nstd,4:nmean
        # logger.debug('feature id: {}, feature:{}'.format(str(feature[0]),(str(nbase),str(nsig),str(nstd),str(nmean))))
    return nfeature


# 0:read_id,1:signal,2:to_pA_shift,3:to_pA_scale,4:sequence,5:stride,6:mv_table,7:num_trimmed,8:to_norm_shift,9:to_norm_scale
# def norm_signal_read_id(signal) -> np.array:
    shift_scale_norm = []
    # signal_norm=[]
    if signal[constants.READ_TO_PA_SCALE] == 0:
        logger.critical("to_pA_scale of read {} is 0").format(signal[0])
    shift_scale_norm = [
        (signal[constants.READ_TO_NORM_SHIFT] / signal[constants.READ_TO_PA_SCALE]) -
        np.float16(signal[constants.READ_TO_PA_SHIFT]),  # SHIFT
        (signal[constants.READ_TO_NORM_SCALE] / signal[constants.READ_TO_PA_SCALE]),  # SCALE
    ]
    # 0:shift,1:scale
    num_trimmed = signal[constants.READ_NUM_TRIMMED]
    # print('num_trimmed:{} and signal:{}'.format(num_trimmed,signal[1]))
    # print('shift:{} and scale:{}'.format(shift_scale_norm[0],shift_scale_norm[1]))
    if shift_scale_norm[constants.READ_SIGNAL] == 0:
        logger.critical("scale of read {} is 0").format(signal[constants.READ_ID])
    if num_trimmed >= 0:
        signal_norm = (
            signal[constants.READ_SIGNAL][num_trimmed:].astype(np.float16) - shift_scale_norm[0]
        ) / shift_scale_norm[1]
    else:
        signal_norm = (
            signal[constants.READ_SIGNAL][:num_trimmed].astype(np.float16) - shift_scale_norm[0]
        ) / shift_scale_norm[1]

    return signal_norm


def scale_signal(signals) -> np.array:
    num_trimmed = signals[constants.READ_NUM_TRIMMED]
    signal_norm = (
        signals[constants.READ_SIGNAL][num_trimmed:].astype(np.float64) - signals[constants.READ_TO_NORM_SHIFT]
    ) / signals[constants.READ_TO_NORM_SCALE]
    return signal_norm


def _normalize_signals(signals, normalize_method="mad"):
    # num_trimmed = signals[constants.READ_NUM_TRIMMED]
    # sig = signals[constants.READ_SIGNAL][num_trimmed:].astype(np.float64)
    if normalize_method == "zscore":
        sshift, sscale = np.mean(signals), np.std(signals)
    elif normalize_method == "mad":
        sshift, sscale = np.median(signals), robust.mad(signals)
    else:
        raise ValueError("")
    norm_signals = (signals - sshift) / sscale
    return np.around(norm_signals, decimals=6)


def caculate_batch_feature_for_each_base(read_q, feature_q, base_num=0, write_batch=10):
    # print("extrac_features process-{} starts".format(os.getpid()))
    logger.info("extrac_features process-{} starts".format(os.getpid()))
    read_num = 0

    while True:
        if read_q.empty():
            time.sleep(2)
            continue
        # lock.acquire()#thread safe
        read_batch = read_q.get()
        # lock.release()
        if read_batch == "kill":
            read_q.put("kill")
            # time.sleep(10)
            break
        read_num += len(read_batch)
        # flag=0
        # if len(read_batch)>1:
        #    flag=1
        #    pos=bar_q.get()
        #    caculate_bar = tqdm(total = len(read_batch), desc='extract_feature', position=pos)
        #    bar_q.put(pos+1)
        # else:
        #    flag=0
        logger.info("read batch size: {}".format(len(read_batch)))
        nfeature = []
        for read_one in read_batch:
            feature = []
            #    if flag == 1:
            #        caculate_bar.update()
            # print(read_one)
            sequence = read_one[constants.READ_SEQ]  # 这个转成np.array内存占用大很多
            stride = read_one[constants.READ_STRIDE]
            movetable = np.array(read_one[constants.READ_MV_TABLE])
            # num_trimmed = read[read_id]['num_trimmed']
            trimed_signals = scale_signal(read_one)
            trimed_signals = _normalize_signals(trimed_signals)  # 筛掉背景信号,norm
            if trimed_signals.size == 0:
                logger.critical(
                    "norm has error, raw data is {}".format(read_one))
                continue
            move_pos = np.append(np.argwhere(
                movetable == 1).flatten(), len(movetable))
            # print(len(move_pos))

            for move_idx in range(len(move_pos) - 1):
                start, end = move_pos[move_idx], move_pos[move_idx + 1]
                signal = trimed_signals[(start * stride): (end * stride)]  # .tolist()
                if signal.size == 0:
                    logger.critical(
                        "signal is empty, it's crazy, read id is {} and base index is".format(
                            read_one[constants.READ_ID], move_idx
                        )
                    )
                    continue
                if True in np.isnan(signal):
                    logger.critical(
                        "signal has nan for read_id:{}".format(read_one[0]))

                try:
                    mean = np.mean(signal)
                    if np.amax(signal) < mean:
                        logger.critical(
                            "ValueERROR: mean greater than max for read_id:{}".format(
                                read_one[constants.READ_ID]
                            )
                        )
                except Exception as e:
                    logger.critical(signal)
                std = np.std(signal.astype(np.float64))  # np.float16会溢出，好像32也不够用了，直接上64吧
                num = end - start

                feature.append(
                    [
                        read_one[constants.READ_ID],
                        signal,
                        np.float16(std),
                        np.float16(mean),
                        np.int16(num * stride),
                        sequence[move_idx],
                    ]
                )
                # 0:read_id,1:signal,2:std,3:mean,4:num,5:base
                # feature[read_id].append({'signal':signal,'std':str(std),'mean':str(mean),'num':int(num*stride),'base':sequence[move_idx]})
            if base_num != 0:
                nfeature.append(_get_neighbord_feature(
                    sequence, feature, base_num))
                logger.info(
                    "extract neigbor features for read_id:{}".format(
                        read_one[constants.READ_ID])
                )
                if len(nfeature) == write_batch:
                    # lock.acquire()
                    feature_q.put(nfeature)
                    # lock.release()
                    nfeature = []
                    while feature_q.qsize() > 50:
                        time.sleep(2)
                        if feature_q.full():
                            logger.error("queue full")

            # feature_q.put(feature)
        if len(nfeature) != 0:
            # lock.acquire()
            feature_q.put(nfeature)
            # lock.release()
            nfeature = []
        # feature_q.append(feature)

        # print("extrac_features process-{} ending, proceed {} read batch".format(os.getpid(), read_num))
    logger.info(
        "extrac_features process-{} ending, proceed {} read".format(
            os.getpid(), read_num
        )
    )
    # if caculate_bar is not None:
    #    caculate_bar.close()
    # pbar.close()


def _prepare_read(read_q, read, batch_size=1000):
    i = 0
    # j=0
    read_batch = []
    for read_one in read:
        read_batch.append(read_one)
        i = i + 1
        # j=j+1
        # if j==40:
        #    break
        if i == batch_size:
            i = 0
            if read_q.full():
                logger.critical("read_q is full")
            read_q.put(read_batch)
            read_batch = []
    if len(read) % batch_size != 0:
        read_q.put(np.array(read_batch, dtype=object))
    # print('total batch number is {}'.format((len(read)-1)//batch_size+1))
    logger.info("total batch number is {}".format(
        (len(read) - 1) // batch_size + 1))
    # return len(read)


def write_feature(read_number, file, feature_q):
    # print("write_process-{} starts".format(os.getpid()))
    logger.info("write_process-{} starts".format(os.getpid()))
    # dataset = []
    # pos=bar_q.get()
    write_feature_bar = tqdm(
        total=read_number,
        desc="extract feature",
        position=0,
        colour="green",
        unit=" read",
    )
    # bar_q.put(pos+1)
    try:
        with open(file, "w") as f:
            while True:
                if feature_q.empty():
                    time.sleep(1)
                    continue
                write_batch = feature_q.get()
                if write_batch == "kill":
                    logger.info(
                        "write_process-{} finished".format(os.getpid()))
                    # time.sleep(10)
                    # np_data = np.array(dataset,dtype=object)
                    # np.save('/home/xiaoyf/methylation/deepsignal/log/data.npy', np_data)
                    # 包含neigbor feature的40条reads保存成npy需要27.87GB，这个开销是无法忍受的
                    # print('write_process-{} finished'.format(os.getpid()))

                    break

                # logger.debug('feature id: {}'.format(str(features[0][0])))
                for read in write_batch:
                    write_feature_bar.update()
                    logger.info(
                        "write process get neigbor features number:{}".format(
                            len(read))
                    )
                    for feature in read:
                        # 0:read_id,1:nbase,2:nsig,3:nstd,4:nmean
                        # #f.write(read_id+'\t')
                        read_id = feature[0]
                        seq = ";".join([",".join([y for y in x])
                                       for x in feature[1]])
                        signal = ";".join(
                            [",".join([str(y) for y in x]) for x in feature[2]]
                        )
                        one_features_str = "\t".join([read_id, seq, signal])
                        f.write(one_features_str + "\n")
                        # np.savetxt(f,np.array("\t".join([read_id,seq,signal])))
                        # dataset.append(feature)
                        # f.write(str(feature[0])+'\t'+str(feature[1])+
                        #        '\t'+str(feature[2])+'\n')

                f.flush()
    except Exception as e:
        logger.critical(
            "error in writing features, this always happend because memory not enough"
        )
        except_type, except_value, except_traceback = sys.exc_info()
        except_file = os.path.split(
            except_traceback.tb_frame.f_code.co_filename)[1]
        exc_dict = {
            "error type": except_type,
            "error imformation": except_value,
            "error file": except_file,
            "error line": except_traceback.tb_lineno,
        }
        print(exc_dict)
    finally:
        write_feature_bar.close()


def extract_feature(read, output_file, nproc=4, batch_size=20, window_size=21):
    start = time.time()
    feature_q = Queue()
    read_q = Queue()
    # bar=Queue()
    # bar.put(0)
    # caculate_batch_feature_pbar = Manager().Queue()
    # write_pbar = Manager().Queue()
    _prepare_read(read_q, read, batch_size)
    read_number = len(read)
    feature_procs = []
    read_q.put("kill")
    # manager = mp.Manager()
    # lock = manager.Lock() #初始化一把锁

    # extract_feature_bar = mp.Process(target=bar_listener, args=(caculate_batch_feature_pbar, "extract_features", 1,))
    # extract_feature_bar.daemon = True
    # extract_feature_bar.start()

    for _ in range(nproc):
        p = mp.Process(
            target=caculate_batch_feature_for_each_base,
            args=(
                read_q,
                feature_q,
                window_size,
            ),
        )
        p.daemon = True
        p.start()
        feature_procs.append(p)

    # write_filename = "/home/xiaoyf/methylation/deepsignal/log/data.npy"

    # write_feature_bar = mp.Process(target=bar_listener, args=(write_pbar, "write_features", 2,))
    # write_feature_bar.daemon = True
    # write_feature_bar.start()
    # tqdm(total = 4000, desc="write_features", position=1)

    p_w = mp.Process(
        target=write_feature,
        args=(
            read_number,
            output_file,
            feature_q,
        ),
    )
    p_w.daemon = True
    p_w.start()
    # with tqdm(total = read_number, desc='extract_feature', position=0) as pbar:
    for p in feature_procs:
        p.join()

    # caculate_bar.close()
    # while True:
    #    flag=0
    #    for p in feature_procs:
    #        if not p.is_alive():
    #            flag+=1
    #    if flag==0:
    #        break
    #    if flag!=0 and not p_w.is_alive():
    #        logger.error("p_w terminate error")
    #        p_w.join()
    #        p_w.start()
    while True:
        flag = 0
        for p in feature_procs:
            if p.is_alive():
                flag += 1
        if flag == 0:
            break
    feature_q.put("kill")
    p_w.join()
    # write_feature_bar.close()

    # extract_feature_bar.join()
    # write_feature_bar.join()
    # print("[main]extract_features costs %.1f seconds.." %(time.time() - start))
    logger.info("[main]extract_features costs %.1f seconds.." %
                (time.time() - start))


if __name__ == "__main__":

    batch_size = args.batch_size
    window_size = args.window_size
    output_file = args.output_file
    log_file = args.log_file
    pod5_path = args.pod5_file
    bam_path = args.bam_file
    nproc = max(mp.cpu_count() - 2, args.nproc)

    read = read_from_pod5_bam(pod5_path, bam_path)
    extract_feature(read, output_file, nproc, batch_size, window_size)
