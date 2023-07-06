import csv
import os
import sys
import argparse
import time, re
import builtins
import numpy as np
import random 
import pickle 
import socket 
import math 
from tqdm import tqdm 
from backbone.select_backbone import select_backbone

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
from torchvision import transforms
import torchvision.utils as vutils

import utils.augmentation as A
import utils.transforms as T
import utils.tensorboard_utils as TB
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils.utils import AverageMeter, write_log, calc_topk_accuracy, calc_mask_accuracy, \
batch_denorm, ProgressMeter, neq_load_customized, save_checkpoint, Logger, FastDataLoader, Cross_dataset
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
from model.pretrain import InfoNCE, UberNCE
from dataset.lmdb_dataset import *


# Text classification using Roberta model from HuggingFace
from transformers import RobertaModel, RobertaForSequenceClassification

checkpoint = 'arxyzan/data2vec-roberta-base'
# this is exactly a roberta model but trained with data2vec
data2vec_roberta = RobertaModel.from_pretrained(checkpoint)
text_classifier = RobertaForSequenceClassification(data2vec_roberta.config)
# assign `data2vec-roberta` weights to the roberta block of the classifier
text_classifier.roberta = data2vec_roberta

import sys
now = os.getcwd()
sys.path.append("D:\Colonpolyps\code\sccl")
# print(os.getcwd())#显示当前路径
import main as sccl
from dataloader.dataloader import augment_loader
from learners.cluster import ClusterLearner
from sccl.utils.logger import statistics_log
from evaluation import prepare_task_input, evaluate_embedding
import time
os.chdir(now)

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='s3d', type=str)
    parser.add_argument('--model', default='infonce', type=str)
    parser.add_argument('--dataset', default='endo240', type=str)
    parser.add_argument('--seq_len', default=32, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=1, type=int, help='number of video blocks')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[3], nargs='*', type=int, help='learning rate schedule (when to drop lr by 1x)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of eval')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--prefix', default='pretrain', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # parallel configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed',default=1, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # for torch.distributed.launch
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)') #修改：原来为2048
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.local_rank != -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'

    ngpus_per_node = torch.cuda.device_count()

    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     assert args.local_rank == -1
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    #     main_worker(args.gpu, ngpus_per_node, args)
    main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0
    args.gpu = gpu

    if args.distributed:
        if args.local_rank != -1: # torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        elif args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args.print = args.gpu == 0
    # suppress printing if not master
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    ### model ###
    print("=> creating {} model with '{}' backbone".format(args.model, args.net))
    if args.model == 'infonce':
        model = InfoNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    elif args.model == 'ubernce':
        model = UberNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    args.num_seq = 2
    print('Re-write num_seq to %d' % args.num_seq)

    args.img_path, args.model_path, args.exp_path = set_path(args)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    device = torch.device("cuda:0")
    model.to(device)


    # ### optimizer ###
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})
    #
    #
    # print('\n===========Check Grad============')
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(name, param.requires_grad)
    # print('=================================\n')
    #
    # optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    ### data ###  
    transform_train = get_transform('train', args)
    train_loader = get_dataloader(get_data(transform_train, 'train', args), 'train', args)
    "define the cluster head for text encoder"

    transform_train_cuda = transforms.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225], channel=1)])
    # n_data = len(train_loader.dataset[0])

    # for idx, x in enumerate(train_loader.dataset[0]):
    #     print(x)
    # print(0)

    print('===================================')

    ### text model and optimizer for video and text ###
    # args for text
    now = os.getcwd()
    sccl_path = "D:\Colonpolyps\code\sccl"
    sys.path.append(sccl_path)

    s_args = sccl.get_args(sys.argv[1:])
    s_args.result_path = sccl_path + '/result'
    s_args.max_iter = 1
    s_args.print_freq = 1

    os.chdir(now)

    text_model, optimizer = sccl.Gemodel_op(s_args, args, model.named_parameters(), train_loader.dataset.dataset2)

    #define the learner for text encoder
    learner = ClusterLearner(text_model, optimizer, s_args.temperature, s_args.base_temperature)

    learner.to(device)

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            try: model_without_ddp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_ddp, state_dict, verbose=True)

            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            try: optimizer.load_state_dict(checkpoint['optimizer'])
            except: print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']

            try: model_without_ddp.load_state_dict(state_dict)
            except: neq_load_customized(model_without_ddp, state_dict, verbose=True)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}', use random init".format(args.pretrain))

    else:
        print("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # tensorboard plot tools
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.train_plotter = TB.PlotterThread(writer_train)

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        _, train_acc = train_one_epoch(train_loader, learner, model, criterion, optimizer, transform_train_cuda, epoch, args, s_args)

        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            # save check_point on rank==0 worker
            if (not args.multiprocessing_distributed and args.rank == 0) \
                or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                is_best = train_acc > best_acc
                best_acc = max(train_acc, best_acc)
                state_dict = model_without_ddp.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration}
                save_checkpoint(save_dict, is_best, gap=args.save_freq,
                    filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch),
                    keep_all='k400' in args.dataset)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
    sys.exit(0)


def training(train_loader, learner, args):
    # print('\n={}/{}=Iterations/Batches'.format(args.max_iter, len(train_loader)))
    t0 = time.time()
    learner.model.train()
    for i in np.arange(args.max_iter + 1):
        try:
            batch = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        feats, _ = prepare_task_input(learner.model, batch, args, is_contrastive=True)

        losses = learner.forward(feats, use_perturbation=args.use_perturbation)
        # print(0)
        # if (args.print_freq > 0) and ((i % args.print_freq == 0) or (i == args.max_iter)):
        #     statistics_log(args.tensorboard, losses=losses, global_step=i)
        #     evaluate_embedding(learner.model, args, i)
        #     learner.model.train()
        # # STOPPING CRITERION (due to some license issue, we still need some time to release the data)
        # you need to implement your own stopping criterion, the one we typically use is
        # diff (cluster_assignment_at_previous_step - cluster_assignment_at_previous_step) / all_data_samples <= criterion
    return losses

def train_one_epoch(data_loader, learner, model, criterion, optimizer, transforms_cuda, epoch, args, s_args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}]'.format(epoch))

    model.train() 

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B,3,args.num_seq,args.seq_len,args.img_dim,args.img_dim)\
        .transpose(1,2).contiguous()

    tic = time.time()
    end = time.time()
    idx = 0

    for input, id_seq in tqdm(data_loader, total=len(data_loader), disable=True):
        idx = idx+1
        data_time.update(time.time() - end)
        input_seq, label = input[0], input[1]
        B = input_seq.size(0)
        input_seq = tr(input_seq.cuda(non_blocking=True))

        if args.model == 'infonce': # 'target' is the index of self
            output, target = model(input_seq)#label是全0？

            #对id data用sccl处理
            #perform the agument_loader function first
            id_seq = augment_loader(train_data=id_seq)
            # device = torch.device("cuda:0")
            # id_seq.to(device)

            loss_id = training(id_seq, learner, s_args)
            loss_id = np.array(loss_id['Instance-CL_loss'].cpu()) + np.array(loss_id['clustering_loss'].cpu()) + np.array(loss_id['local_consistency_loss'])
            loss_id = torch.tensor(loss_id, requires_grad=True)
            # print(sccl.run(s_args, id_seq))

            # loss_id = 0.0
            # loss_vi = criterion(output, target)#output包含了正对，负对
            loss_vi = 0.0
            loss = 0.1*loss_id + loss_vi
            top1, top5 = calc_topk_accuracy(output, target, (1,5))
        
        if args.model == 'ubernce': # 'target' is the binary mask
            label = label.cuda(non_blocking=True)
            output, target = model(input_seq, label)
            # optimize all positive pairs, compute the mean for num_pos and for batch_size 
            loss = - (F.log_softmax(output, dim=1) * target).sum(1) / target.sum(1)
            loss = loss.mean()
            top1, top5 = calc_mask_accuracy(output, target, (1,5))

        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        losses.update(loss.item(), B)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        
        progress.display(idx)

        if idx % args.print_freq == 0:
            if args.print:
                args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
                args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)


        #write into csv
        #create csv file first
        if idx==0:
            with open(r'result.csv', 'w') as f:
                f.truncate()
                writer = csv.writer(f)
                writer.writerow(["index", "losses", "top1", "top5"])
        #write data
        with open(r'result.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, loss.item(), top1.item(), top5.item()])
        
        args.iteration += 1

    print('({gpu:1d})Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), gpu=args.rank, t=time.time()-tic))
    
    if args.print:
        args.train_plotter.add_data('global/loss', losses.avg, epoch)
        args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
        args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)


    return losses.avg, top1_meter.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    # stepwise lr schedule
    for milestone in args.schedule:
        if epoch % milestone == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5


def get_transform(mode, args):
    null_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=args.seq_len, bottom_area=0.2),
        A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
        A.ToTensor(),
    ])

    base_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=args.seq_len, bottom_area=0.2),
        transforms.RandomApply([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, consistent=False, seq_len=args.seq_len)
            ], p=0.8),
        A.RandomGray(p=0.2, seq_len=args.seq_len),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=args.seq_len)], p=0.5),
        # lessaug版本将以上四行暂时注释
        A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
        A.ToTensor(),
    ])

    # oneclip: temporally take one clip, random augment twice
    # twoclip: temporally take two clips, random augment for each
    # merge oneclip & twoclip transforms with 50%/50% probability
    transform = A.TransformController(
                    [A.TwoClipTransform(base_transform, null_transform, seq_len=args.seq_len, p=0.3),
                     A.OneClipTransform(base_transform, null_transform, seq_len=args.seq_len)],
                    weights=[0.5,0.5])
    # print(transform)
    return transform 

def get_data(transform, mode, args):
    print('Loading data for "%s" mode' % mode)

    if args.dataset == 'ucf101-2clip':
        dataset = UCF101LMDB_2CLIP(mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)
    elif args.dataset == 'ucf101-f-2clip':
        dataset = UCF101Flow_LMDB_2CLIP(mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)

    elif args.dataset == 'k400-2clip': 
        dataset = K400_LMDB_2CLIP(mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)
    elif args.dataset == 'k400-f-2clip': 
        dataset = K400_Flow_LMDB_2CLIP(mode=mode, transform=transform, 
            num_frames=args.seq_len, ds=args.ds, return_label=True)

    elif args.dataset == 'endo240':
        dataset = ENDO240LMDB_2CLIP(mode=mode, transform=transform,
                                   num_frames=args.seq_len, ds=args.ds, return_label=True)

    id_dataset = list(dataset.get_video_id.keys())
    new_id_dataset = find_ID(id_dataset)
    return dataset, new_id_dataset

def find_ID(id):
    with open("./dataset/eda_colon.csv", "r", encoding='utf-8') as f:
        data = csv.reader(f)
        rows = [row for row in data][1:]
        for row in rows:
            name = row[0]
            # text = row[1]
            # text1 = row[2]
            # text2 = row[3]
            # label = row[4]
            ifo = row[1:]

            #find the name in the id_dataset and repalce it with text information
            for i in range(len(id)):
                if name in id[i]:
                    id[i] = ifo
            # print(row)

    return id

def train_collate_fn(batch):
    img, id = zip(*batch)
    return img, id,

def get_dataloader(dataset, mode, args):
    print('Creating data loaders for "%s" mode' % mode)
    dataset = Cross_dataset(dataset1=dataset[0], dataset2=dataset[1])
    train_sampler = data.distributed.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)#这里可以读取id
    if mode == 'train':
        data_loader = FastDataLoader(
            dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True,  sampler=train_sampler, drop_last=True)
    else:
        raise NotImplementedError
    ('"%s" dataset has size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log-{args.prefix}/{args.name_prefix}{args.model}_k{args.moco_k}_{args.dataset}-{args.img_dim}_{args.net}_\
bs{args.batch_size}_lr{args.lr}_seq{args.num_seq}_len{args.seq_len}_ds{args.ds}_{0}'.format(
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    # exp_path = 'log-pretrain/infonce_k65536_endo240-128_s3d_bs16_lr0.001_seq2_len32_ds1_lessaug' #修改
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): 
        if args.distributed and args.gpu == 0:
            os.makedirs(img_path)
    if not os.path.exists(model_path): 
        if args.distributed and args.gpu == 0:
            os.makedirs(model_path)
    return img_path, model_path, exp_path

if __name__ == '__main__':
    '''
    Three ways to run (recommend first one for simplicity):
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank
       
    2. CUDA_VISIBLE_DEVICES=0,1 python main_nce.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. using SLURM scheduler
    '''
    args = parse_args()
    main(args)