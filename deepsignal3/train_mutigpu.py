
from utils.utils import parse_args
import os
import sys
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn import metrics
import re
from tqdm import tqdm
from dataloader import SignalFeaData2, clear_linecache
import torch
from model import CapsNet, CapsuleLoss
from torch.utils.data import sampler
import signal
import logging
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

args = parse_args()
signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
logger = logging.getLogger("train_logger")
logger.setLevel(logging.DEBUG)
processdata_log = logging.FileHandler(
    args.log_file, "a", encoding="utf-8"
)
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s"
)
processdata_log.setFormatter(formatter)
# 加载文件到logger对象中
logger.addHandler(processdata_log)
use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=6)
    print(output)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def cleanup():
    dist.destroy_process_group()

def checkpoint(model, gpu, model_save_path):
    """Saves the model in master process and loads it everywhere else.

    Args:
        model: the model to save
        gpu: the device identifier
        model_save_path:
    Returns:
        model: the loaded model
    """
    if gpu == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.module.state_dict(), model_save_path)

    # use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    model.module.load_state_dict(
        torch.load(model_save_path, map_location=map_location))

def train_worker(local_rank, global_world_size, args):
    global_rank = args.node_rank * args.ngpus_per_node + local_rank

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=global_world_size,
        rank=global_rank,
    )
    print('world_size', dist.get_world_size())  # 打印当前进程数
    sys.stderr.write("training_process-{} [init] == local rank: {}, global rank: {} ==\n".format(os.getpid(),
                                                                                                local_rank,
                                                                                                global_rank))
    
    sys.stderr.write("training_process-{} reading data..\n".format(os.getpid()))
    train_dataset = SignalFeaData2(args.train_file, args.target_chr)

    split_num = int(len(train_dataset) * 0.8)
    index_list = list(range(len(train_dataset)))
    train_idx, val_idx = index_list[:split_num], index_list[split_num:]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_idx,
                                                                        shuffle=True)#sampler.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_idx,
                                                                        shuffle=True)#sampler.SubsetRandomSampler(val_idx)
    logger.info('trainning and verifying batch size is {}'.format(args.batch_size))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=2,pin_memory=True,
        shuffle=False,
        persistent_workers=True,prefetch_factor=2
    )
    total_step = len(train_loader)
    valid_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=2,pin_memory=True,
        shuffle=False,
        persistent_workers=True,prefetch_factor=2
    )

    if global_rank == 0 or args.epoch_sync:
        model_dir = args.model_dir
        if model_dir != "/":
            model_dir = os.path.abspath(model_dir).rstrip("/")
            if local_rank == 0:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                else:
                    model_regex = re.compile(
                        r"" + "\\.b\\d+_s\\d+_epoch\\d+\\.ckpt*"
                    )
                    for mfile in os.listdir(model_dir):
                        if model_regex.match(mfile):
                            os.remove(model_dir + "/" + mfile)
                model_dir += "/"
    model = CapsNet(device=local_rank)#
    dist.barrier()
    #if use_cuda:
    #    model = model.cuda(local_rank, non_blocking=True)
    model = model.cuda(local_rank)
    # DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                #find_unused_parameters=True
                )
    #model = torch.nn.DataParallel(model)
    #freeze_layers_prefix = ("embed", "lstm_seq","lstm_sig","primary_layer","caps_layer")
    #for name, param in model.named_parameters():
        #print(name, param.shape)
    #    if name.split('.')[0] in freeze_layers_prefix:
    #        param.requires_grad = False
    
    criterion = CapsuleLoss(device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    if args.lr_scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay,
                                      patience=args.lr_patience, verbose=True)
    else:
        raise ValueError("--lr_scheduler is not right!")

    curr_best_accuracy = 0

    if args.init_model is not None:
        sys.stderr.write("training_process-{} loading pre-trained model: {}\n".format(os.getpid(), args.init_model))
        para_dict = torch.load(args.init_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)
    
    model.train()
    #prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #schedule = torch.profiler.schedule(
        #    skip_first=1,
        #    wait=0,
        #    warmup=2,
        #    active=1
        #), 
        #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')#trace_handler
     #)
    for epoch in range(args.max_epoch_num):
        train_loader.sampler.set_epoch(epoch)
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        estart= time.time()
        #loop = tqdm(enumerate(train_loader), total =len(train_loader))
        data_iter = tqdm(train_loader,
                            desc="epoch %d" % (epoch),
                            total=len(train_loader),
                            bar_format="{l_bar}{r_bar}")
        for i, sfeatures in enumerate(data_iter):
            (seq, sig, labels) = sfeatures
            if use_cuda:
                seq = seq.cuda(local_rank, non_blocking=True)
                sig = sig.cuda(local_rank, non_blocking=True)
                labels = labels.cuda(local_rank, non_blocking=True)
            outputs, logits = model(seq, sig)
            loss = criterion(outputs, labels)
            # print(loss)
            logger.debug(loss.detach().item())
            tlosses.append(loss.detach().item())

            # Backward and optimize
            optimizer.zero_grad()
            #with torch.profiler.record_function("backward"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        #    prof.step()
            if global_rank == 0 and ((i + 1) % args.step_interval == 0 or (i + 1) == total_step):
                
                time_cost = time.time() - start
                logger.info(
                    "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                    "Time: {:.2f}s".format(
                        epoch + 1,
                        args.max_epoch_num,
                        i + 1,
                        total_step,
                        np.mean(tlosses),
                        time_cost,
                    )
                )
                tlosses = []
                start = time.time()
                sys.stdout.flush()
                
        model.eval()
        with torch.no_grad():
            vlosses, vlabels_total, vpredicted_total = [], [], []
            v_meanloss = 10000        
            for vi, vsfeatures in enumerate(valid_loader):#tqdm(valid_loader):
                (
                    vseq,
                    vsig,
                    vlabels,
                ) = vsfeatures
                if use_cuda:
                    vseq = vseq.cuda(local_rank, non_blocking=True)
                    vsig = vsig.cuda(local_rank, non_blocking=True)
                    vlabels = vlabels.cuda(local_rank, non_blocking=True)
                voutputs, vlogits = model(vseq, vsig)
                vloss = criterion(voutputs, vlabels)
                dist.barrier()
                vloss = reduce_mean(vloss, global_world_size)

                _, vpredicted = torch.max(vlogits.data, 1)
                if use_cuda:
                   vpredicted = vpredicted.cpu()
                   vlabels = vlabels.cpu()
                # print(vpredicted)
                vlosses.append(vloss.item())
                vlabels_total += vlabels.tolist()
                vpredicted_total += vpredicted.tolist()
            v_accuracy = metrics.accuracy_score(
                vlabels_total, vpredicted_total
            )
            v_precision = metrics.precision_score(
                vlabels_total, vpredicted_total
            )
            v_recall = metrics.recall_score(vlabels_total, vpredicted_total)
            v_meanloss = np.mean(vlosses)
            if v_accuracy > curr_best_accuracy_epoch:
                curr_best_accuracy_epoch = v_accuracy
                if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                    #torch.save(
                    #    #model.module.state_dict(),
                    #    model.state_dict(),
                    #    model_dir
                    #    + args.target_chr + ".epoch{}.ckpt".format(
                    #        epoch
                    #    ),
                    #)
                    if global_rank == 0:
                        # model.state_dict() or model.module.state_dict()?
                        torch.save(model.module.state_dict(),
                            model_dir  + args.target_chr + ".epoch{}.ckpt".format(
                            epoch
                        ))
                    if curr_best_accuracy_epoch > curr_best_accuracy:
                        curr_best_accuracy = curr_best_accuracy_epoch
                        no_best_model = False
                time_cost = time.time() - estart
                if global_rank == 0:
                    logger.info(
                        "Epoch [{}/{}], Step [{}/{}]; "
                        "ValidLoss: {:.4f}, "
                        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                        "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                            epoch + 1,
                            args.max_epoch_num,
                            i + 1,
                            total_step,
                            np.mean(vlosses),
                            v_accuracy,
                            v_precision,
                            v_recall,
                            curr_best_accuracy_epoch,
                            time_cost,
                        )
                    )
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "ValidLoss": np.mean(vlosses),
                    "Accuracy": v_accuracy,
                }
                data_iter.write(str(post_fix))
                sys.stderr.flush()
                #loop.set_description(f'Epoch [{epoch}/{args.max_epoch_num}]')
                #loop.set_postfix(loss = loss.item(),acc = v_accuracy,precision=v_precision,recall=v_recall)
                    
        model.train()
        if args.epoch_sync:
            sync_ckpt = model_dir + args.model_type + \
                        '.epoch_sync_node{}.target_{}_epoch{}.ckpt'.format(args.node_rank, args.target_chr, epoch + 1)
            checkpoint(model, local_rank, sync_ckpt)
        if args.lr_scheduler == "ReduceLROnPlateau":
            lr_reduce_metric = v_meanloss
            scheduler.step(lr_reduce_metric)
        else:
            scheduler.step()
        if no_best_model and epoch >= args.min_epoch_num - 1:
            logger.info("early stop!")
            break
    endtime = time.time()  
    if global_rank == 0:
        logger.info(
            "[main] train costs {} seconds, "
            "best accuracy: {}".format(endtime - args.total_start, curr_best_accuracy)
        )
    clear_linecache()
    cleanup()
    
    #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    #logger.info(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

def train(args):
    logger.info("[main]train_multigpu starts")
    args.total_start = time.time()
    print(torch.cuda.device_count())  # 打印gpu数量
    

    torch.manual_seed(args.tseed)
    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

    if use_cuda:
        logger.info("GPU is available!")
    else:
        raise RuntimeError("No GPU is available!")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available!")

    if torch.cuda.device_count() < args.ngpus_per_node:
        raise RuntimeError("There are not enough gpus, has {}, request {}.".format(torch.cuda.device_count(),
                                                                                   args.ngpus_per_node))

    global_world_size = args.ngpus_per_node * args.nodes
    mp.spawn(train_worker, nprocs=args.ngpus_per_node, args=(global_world_size, args))

    endtime = time.time()
    clear_linecache()
    logger.info("[main]train_multigpu costs {:.1f} seconds".format(endtime - args.total_start))


if __name__ == "__main__":
    logger.info('use gpu: {}'.format(use_cuda))
        
    
    train(args)
    
