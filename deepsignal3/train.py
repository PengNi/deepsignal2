
from utils.utils import parse_args
import os
import sys
import numpy as np
from torch.optim.lr_scheduler import StepLR
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

if __name__ == "__main__":
    logger.info('use gpu: {}'.format(use_cuda))
    total_start = time.time()
    train_dataset = SignalFeaData2(args.train_file, args.target_chr)

    split_num = int(len(train_dataset) * 0.8)
    index_list = list(range(len(train_dataset)))
    train_idx, val_idx = index_list[:split_num], index_list[split_num:]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(val_idx)
    logger.info('trainning and verifying batch size is {}'.format(args.batch_size))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4,pin_memory=True,
        persistent_workers=True,prefetch_factor=2
    )
    total_step = len(train_loader)
    valid_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=2,pin_memory=True,
        persistent_workers=True,prefetch_factor=2
    )
    model = CapsNet()#torch.nn.DataParallel(CapsNet())
    #freeze_layers_prefix = ("embed", "lstm_seq","lstm_sig","primary_layer","caps_layer")
    #for name, param in model.named_parameters():
        #print(name, param.shape)
    #    if name.split('.')[0] in freeze_layers_prefix:
    #        param.requires_grad = False
    if use_cuda:
        model = model.cuda()
    criterion = CapsuleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    curr_best_accuracy = 0
    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
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
    model.train()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule = torch.profiler.schedule(
            skip_first=1,
            wait=0,
            warmup=2,
            active=1
        ), 
        on_trace_ready=trace_handler
     ) as prof:
        for epoch in range(args.max_epoch_num):
            curr_best_accuracy_epoch = 0
            no_best_model = True
            tlosses = []
            start = time.time()
            loop = tqdm(enumerate(train_loader), total =len(train_loader))
            for i, sfeatures in loop:#tqdm(enumerate(train_loader),total=len(train_loader),leave = True):
                (seq, sig, labels) = sfeatures
                if use_cuda:
                    seq = seq.cuda()
                    sig = sig.cuda()
                    labels = labels.cuda()
                outputs, logits = model(seq, sig)
                loss = criterion(outputs, labels)
                # print(loss)
                logger.debug(loss.detach().item())
                tlosses.append(loss.detach().item())

                # Backward and optimize
                optimizer.zero_grad()
                with torch.profiler.record_function("backward"):
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                prof.step()
                if (i + 1) % args.step_interval == 0 or i == total_step - 1:
                    model.eval()
                    with torch.no_grad():
                        vlosses, vlabels_total, vpredicted_total = [], [], []
                        for vi, vsfeatures in tqdm(enumerate(valid_loader)):
                            (
                                vseq,
                                vsig,
                                vlabels,
                            ) = vsfeatures
                            if use_cuda:
                                vseq = vseq.cuda()
                                vsig = vsig.cuda()
                                vlabels = vlabels.cuda()
                            voutputs, vlogits = model(vseq, vsig)
                            vloss = criterion(voutputs, vlabels)

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
                        if v_accuracy > curr_best_accuracy_epoch:
                            curr_best_accuracy_epoch = v_accuracy
                            if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                                torch.save(
                                    #model.module.state_dict(),
                                    model.state_dict(),
                                    model_dir
                                    + args.target_chr + ".epoch{}.ckpt".format(
                                        epoch
                                    ),
                                )
                                if curr_best_accuracy_epoch > curr_best_accuracy:
                                    curr_best_accuracy = curr_best_accuracy_epoch
                                    no_best_model = False
                        time_cost = time.time() - start
                        logger.info(
                            "Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; "
                            "ValidLoss: {:.4f}, "
                            "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
                            "curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s".format(
                                epoch + 1,
                                args.max_epoch_num,
                                i + 1,
                                total_step,
                                np.mean(tlosses),
                                np.mean(vlosses),
                                v_accuracy,
                                v_precision,
                                v_recall,
                                curr_best_accuracy_epoch,
                                time_cost,
                            )
                        )
                        loop.set_description(f'Epoch [{epoch}/{args.max_epoch_num}]')
                        loop.set_postfix(loss = loss.item(),acc = v_accuracy,precision=v_precision,recall=v_recall)
                        tlosses = []
                        start = time.time()
                        sys.stdout.flush()
                    model.train()
            scheduler.step()
            if no_best_model and epoch >= args.min_epoch_num - 1:
                logger.info("early stop!")
                break
    
    
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    logger.info(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    
    endtime = time.time()
    clear_linecache()
    logger.info(
        "[main] train costs {} seconds, "
        "best accuracy: {}".format(endtime - total_start, curr_best_accuracy)
    )
