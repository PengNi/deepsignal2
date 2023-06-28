import multiprocessing as mp
import numpy as np
import pod5 as p5
import pysam
from multiprocessing import Queue#,Manager,Pool
import time
import os
import signal
import logging
from tqdm import tqdm
from multiprocessing import Pool
from utils.utils import parse_args
from utils.log import get_logger,init_logger

signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
LOGGER = get_logger()

def extract_signal_from_pod5(pod5_path)-> list:
    signals=[]
    with p5.Reader(pod5_path) as reader:
        for read_record in reader.reads():
            #signals[str(read_record.read_id)] = {'signal':read_record.signal,'shift':read_record.calibration.offset,'scale':read_record.calibration.scale}#不加str会变成UUID，很奇怪
            signals.append([str(read_record.read_id),read_record.signal,read_record.calibration.offset,read_record.calibration.scale])
            #0:read_id,1:signal,2:shift,3:scale
    return signals
def extract_move_from_bam(bam_path)-> list:
    seq_move=[]
    bamfile = pysam.AlignmentFile(bam_path, "rb",check_sq=False)
    try:
        for read in bamfile.fetch(until_eof=True):#暂时不使用索引，使用返回是空值
            #print(read.query_name)
            tags=dict(read.tags)
            mv_tag=tags['mv']
            ts_tag=tags['ts']
            sm_tag=tags["sm"]
            sd_tag=tags["sd"]
            #read.update({read.query_name:{"sequence":read.query_sequence,"stride":mv_tag[0],"mv_table":np.array(mv_tag[1:]),"num_trimmed":ts_tag,"shift":sm_tag,"scale":sd_tag}})
            seq_move.append([read.query_name,read.query_sequence,mv_tag[0],np.array(mv_tag[1:]),ts_tag,sm_tag,sd_tag])
    except ValueError:
        print('bam don\'t has index')
        for read in bamfile.fetch(until_eof=True,multiple_iterators=False):
            tags=dict(read.tags)
            mv_tag=tags['mv']
            ts_tag=tags['ts']
            sm_tag=tags["sm"]
            sd_tag=tags["sd"]
            seq_move.append([read.query_name,read.query_sequence,mv_tag[0],np.array(mv_tag[1:]),ts_tag,sm_tag,sd_tag])
            #0:read_id,1:sequence,2:stride,3:mv_table,4:num_trimmed,5:to_norm_shift,6:to_norm_scale
            #read[read.query_name] = {"sequence":read.query_sequence,"stride":mv_tag[0],"mv_table":np.array(mv_tag[1:]),"num_trimmed":ts_tag,"shift":sm_tag,"scale":sd_tag}
    return seq_move
def read_from_pod5_bam(pod5_path,bam_path,read_id=None)-> list:
    read=[]
    signal = extract_signal_from_pod5(pod5_path)
    seq_move = extract_move_from_bam(bam_path)
    if read_id is not None:
        for i in range(len(seq_move)):
            if seq_move[i][0]==read_id:
                if seq_move[i][1] is not None:
                    for j in range(len(signal)):
                        if signal[j][0]==seq_move[i][0]:
                            read.append([signal[j][0],signal[j][1],signal[j][2],signal[j][3],
                            seq_move[i][1],seq_move[i][2],seq_move[i][3],seq_move[i][4],seq_move[i][5],seq_move[i][6]])
        
    else:
        for i in range(len(seq_move)):
            if seq_move[i][1] is not None:
                for j in range(len(signal)):
                    if signal[j][0]==seq_move[i][0]:
                        read.append([signal[j][0],signal[j][1],signal[j][2],signal[j][3],
                        seq_move[i][1],seq_move[i][2],seq_move[i][3],seq_move[i][4],seq_move[i][5],seq_move[i][6]])
                #0:read_id,1:signal,2:to_pA_shift,3:to_pA_scale,4:sequence,5:stride,6:mv_table,7:num_trimmed,8:to_norm_shift,9:to_norm_scale
                    
                
    return read

#0:read_id,1:signal,2:std,3:mean,4:num,5:base 
def _get_neighbord_feature(feature,base_num):
    #数据预处理主要速度瓶颈，同样的reads数，不运行这个函数大概快了十倍，从二十多分钟减到两分钟
    nfeature=[]
    windows_size=base_num-1//2
    for i in range(len(feature)):
        nbase=[]
        nstd=[]
        nmean=[]
        nsig=[]
        if i<windows_size:                   
            if i!=0:
                for k in range(i):
                    nbase=nbase+list(feature[k][5])*feature[k][4]
                    nstd=nstd+list(feature[k][2])*feature[k][4]
                    nmean=nmean+list(feature[k][3])*feature[k][4]
                    nsig=nsig+feature[k][1]
            nbase=nbase+list(feature[i][5])*(windows_size-i)*feature[i][4]
            nbase=nbase+list(feature[i][5])*feature[i][4]
            nstd=nstd+list(feature[i][2])*(windows_size-i)*feature[i][4]
            nstd=nstd+list(feature[i][2])*feature[i][4]
            nmean=nmean+list(feature[i][3])*(windows_size-i)*feature[i][4]
            nmean=nmean+list(feature[i][3])*feature[i][4]
            nsig=nsig+feature[i][1]*(windows_size-i)
            nsig=nsig+feature[i][1]
            for k in range(i,i+windows_size):
                nbase=nbase+list(feature[k][5])*feature[k][4]
                nstd=nbase+list(feature[k][2])*feature[k][4]
                nmean=nbase+list(feature[k][3])*feature[k][4]
                nsig=nsig+feature[k][1]
        elif (len(feature[i])-1)-i<windows_size:
            for k in range(i-windows_size,i):
                nbase=nbase+list(feature[k][5])*feature[k][4]
                nstd=nstd+list(feature[k][2])*feature[k][4]
                nmean=nmean+list(feature[k][3])*feature[k][4]
                nsig=nsig+feature[k][1]
            nbase=nbase+list(feature[i][5])*feature[i][4]
            nstd=nstd+list(feature[i][2])*feature[i][4]
            nmean=nmean+list(feature[i][3])*feature[i][4]        
            nsig=nsig+feature[i][1]                   
            if i!=len(feature[i])-1:
                for k in range(i,len(feature[i])-1):
                    nbase=nbase+list(feature[k][5])*feature[k][4]
                    nstd=nstd+list(feature[k][2])*feature[k][4]
                    nmean=nmean+list(feature[k][3])*feature[k][4]
                    nsig=nsig+feature[k][1]
            nbase=nbase+list(feature[i][5])*(windows_size-((len(feature[i])-1)-i))*feature[i][4]
            nstd=nstd+list(feature[i][2])*(windows_size-((len(feature[i])-1)-i))*feature[i][4]
            nmean=nmean+list(feature[i][3])*(windows_size-((len(feature[i])-1)-i))*feature[i][4]
            nsig=nsig+feature[i][1]*(windows_size-((len(feature[i])-1)-i))
        else:
            for k in range(i-windows_size,i):
                nbase=nbase+list(feature[k][5])*feature[k][4]
                nstd=nstd+list(feature[k][2])*feature[k][4]
                nmean=nmean+list(feature[k][3])*feature[k][4]
                nsig=nsig+feature[k][1]
            nbase=nbase+list(feature[i][5])*feature[i][4]
            nstd=nstd+list(feature[i][2])*feature[i][4]
            nmean=nmean+list(feature[i][3])*feature[i][4]
            nsig=nsig+feature[i][1]
            for k in range(i,i+windows_size):
                nbase=nbase+list(feature[k][5])*feature[k][4]
                nstd=nstd+list(feature[k][2])*feature[k][4]
                nmean=nmean+list(feature[k][3])*feature[k][4]
                nsig=nsig+feature[k][1]
        #feature[read_id][i].update({'nbase':nbase,'nsig':nsig,'nstd':nstd,'nmean':nmean})
        nfeature.append([feature[i][0],nbase,nsig,nstd,nmean])
        
        #0:read_id,1:nbase,2:nsig,3:nstd,4:nmean
        #LOGGER.debug('feature id: {}, feature:{}'.format(str(feature[0]),(str(nbase),str(nsig),str(nstd),str(nmean))))
    return nfeature
        
#0:read_id,1:signal,2:to_pA_shift,3:to_pA_scale,4:sequence,5:stride,6:mv_table,7:num_trimmed,8:to_norm_shift,9:to_norm_scale
def norm_signal_read_id(signal):
    shift_scale_norm=[]
    signal_norm=[]
    shift_scale_norm=[(signal[8]/signal[3])-signal[2],(signal[9]/signal[3])]
    #0:shift,1:scale

    num_trimmed=signal[7]
    #print('num_trimmed:{} and signal:{}'.format(num_trimmed,signal[1]))
    #print('shift:{} and scale:{}'.format(shift_scale_norm[0],shift_scale_norm[1]))
    signal_norm=(signal[1][num_trimmed:] - shift_scale_norm[0]) / shift_scale_norm[1]        
    return signal_norm

def caculate_batch_feature_for_each_base(bar_q,read_q,feature_q,base_num=0):
    print("extrac_features process-{} starts".format(os.getpid()))
    LOGGER.info("extrac_features process-{} starts".format(os.getpid()))
    read_num = 0
    
    while True:
        if read_q.empty():
            time.sleep(10)
        read_batch=read_q.get()
        if read_batch == "kill":
            read_q.put("kill")
            break
        read_num += len(read_batch)
        flag=0
        if len(read_batch)>1:
            flag=1
            pos=bar_q.get()
            caculate_bar = tqdm(total = len(read_batch), desc='extract_feature', position=pos)
            bar_q.put(pos+1)
        else:
            flag=0
                
        for read_one in read_batch:
            feature=[]
            if flag == 1:
                caculate_bar.update()
            #print(read_one)            
            sequence = read_one[4]
            stride = read_one[5]
            movetable = read_one[6]           
            #num_trimmed = read[read_id]['num_trimmed']
            trimed_signals = norm_signal_read_id(read_one)#筛掉背景信号,norm
            move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
            #print(len(move_pos))
            for move_idx in range(len(move_pos) - 1):
                start, end = move_pos[move_idx], move_pos[move_idx + 1]
                signal=trimed_signals[(start * stride):(end * stride)].tolist()
                mean=np.mean(signal)
                std=np.std(signal)
                num=end-start
                #print(move_idx)
                feature.append([read_one[0],signal,str(std),str(mean),int(num*stride),sequence[move_idx]])
                #0:read_id,1:signal,2:std,3:mean,4:num,5:base        
                #feature[read_id].append({'signal':signal,'std':str(std),'mean':str(mean),'num':int(num*stride),'base':sequence[move_idx]})
            #if base_num!=0:
            #    nfeature=_get_neighbord_feature(feature,base_num)
            #    LOGGER.debug("extract neigbor features for read_id:{}".format(read_one[0]))
            #    feature_q.put(nfeature)
            feature_q.put(feature)
        #feature_q.append(feature)
        
        #print("extrac_features process-{} ending, proceed {} read batch".format(os.getpid(), read_num))
        LOGGER.info("extrac_features process-{} ending, proceed {} read batch".format(os.getpid(), read_num))
        if caculate_bar is not None:
            caculate_bar.close()
    #pbar.close()
        

def _prepare_read(read,batch_size=1000):
    i=0
    #j=0
    read_batch=[]
    for read_one in read:
        read_batch.append(read_one)
        i=i+1
        #j=j+1
        #if j==40:
        #    break
        if i==batch_size:
            i=0
            yield read_batch
            read_batch=[]
    LOGGER.info('total batch number is {}'.format((len(read)-1)//batch_size+1))
    yield read_batch


def write_feature(read_number,bar_q,file,feature_q):
    #print("write_process-{} starts".format(os.getpid()))
    LOGGER.info("write_process-{} starts".format(os.getpid()))
    dataset=[]
    pos=bar_q.get()
    write_feature_bar = tqdm(total = read_number, desc='write_feature', position=pos,colour='green')
    bar_q.put(pos+1)
    try:
        
        while True:
            if feature_q.empty():
                time.sleep(10)
                continue
            features = feature_q.get()
            if features == "kill":
                LOGGER.info('write_process-{} finished'.format(os.getpid()))    
                np_data = np.array(dataset)
                np.save(file, np_data)
                #包含neigbor feature的40条reads保存成npy需要27.87GB，这个开销是无法忍受的
                #不包含neigbor feature的40条reads保存成npy仅需要30.34MB
                #print('write_process-{} finished'.format(os.getpid()))                
                break
            LOGGER.info('write process get bases number:{}'.format(len(features)))
            #LOGGER.debug('feature id: {}'.format(str(features[0][0])))
            for feature in features:
                #0:read_id,1:nbase,2:nsig,3:nstd,4:nmean
                #f.write(read_id+'\t')
                write_feature_bar.update()
                dataset.append(feature)                
                #f.write(str(feature[1])+'\t'+str(feature[2])+
                #            '\t'+str(feature[3])+'\t'+str(feature[4])+'\n')
                    
            
    except Exception as e:
        LOGGER.error('error in writing features')
        print(e)
    finally:
        write_feature_bar.close()
    #write_pbar.close()
            
def bar_listener(p_bar,desc='',position=1,number=4000):
    bar = tqdm(total = number, desc=desc, position=position)
    for item in iter(p_bar.get, None):
        bar.update(item)

def extract_feature(read,output_file,nproc = 4,batch_size=20):
    start = time.time()
    feature_q = Queue()
    #read_q=Queue()
    bar=Queue()
    bar.put(0)
    #caculate_batch_feature_pbar = Manager().Queue()
    #write_pbar = Manager().Queue()
    _prepare_read(read,batch_size)
    read_number=len(read)
    feature_procs=[]  
    #read_q.put("kill")
    
    #extract_feature_bar = mp.Process(target=bar_listener, args=(caculate_batch_feature_pbar, "extract_features", 1,))
    #extract_feature_bar.daemon = True
    #extract_feature_bar.start()
    
    pool = Pool(nproc)
    #with Pool(nproc) as p:
    #    p.map(caculate_batch_feature_for_each_base, _prepare_read)
    tqdm(pool.imap(caculate_batch_feature_for_each_base, _prepare_read), total=read_number, desc='extract_features')    
        
               
    
    write_filename=output_file
    
    #write_feature_bar = mp.Process(target=bar_listener, args=(write_pbar, "write_features", 2,))
    #write_feature_bar.daemon = True
    #write_feature_bar.start()
    #tqdm(total = 4000, desc="write_features", position=1)
    
    p_w = mp.Process(target=write_feature, args=(read_number,bar,write_filename,feature_q,))
    p_w.daemon = False
    p_w.start()
    #with tqdm(total = read_number, desc='extract_feature', position=0) as pbar:
    for p in feature_procs:
        p.join()
    
    #caculate_bar.close()
    feature_q.put("kill")
    p_w.join()
    #write_feature_bar.close()
    
    #extract_feature_bar.join()
    #write_feature_bar.join()
    #print("[main]extract_features costs %.1f seconds.." %(time.time() - start))
    LOGGER.info("[main]extract_features costs %.1f seconds.." %(time.time() - start))

if __name__ == '__main__':
    args = parse_args()
    init_logger(args.log_filename)
    batch_size = args.batch_size
    window_size = args.window_size
    output_file = args.output_file
    log_file = args.log_file
    pod5_path = args.pod5_file
    bam_path = args.bam_file
    nproc = args.nproc
    
    read=read_from_pod5_bam(pod5_path,bam_path)
    extract_feature(read,output_file,nproc,batch_size)

