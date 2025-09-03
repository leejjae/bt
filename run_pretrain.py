import warnings
warnings.filterwarnings("ignore")


import argparse
import random
import json
import math
import os
import sys
import time

from tqdm import tqdm
from pathlib import Path
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.amp import autocast, GradScaler

import torch
import numpy as np

from dataset import DataManager, TransformCL, TwoViewDataset
from model import build_model

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=5, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--src_dataset', type=str, required=True,
                    choices=['mnist','usps','svhn','cifar','cifar10c','cifarv2','cinic'])
parser.add_argument('--tgt_dataset', type=str, required=True,
                    choices=['mnist','usps','svhn','cifar','cifar10c','cifarv2','cinic'])

parser.add_argument('--src_prior', type=float, default=0.5)
parser.add_argument('--tgt_prior', type=float, default=0.5)
parser.add_argument('--arch', type=str, default='lenet')  
parser.add_argument('--feat_dim', type=int, default=None)


def main():
    args = parser.parse_args()
    args.rank = 0
    set_seed(args.seed)
    
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pair_name = f"{args.src_dataset}_{args.tgt_dataset}"
    seed_dir  = args.checkpoint_dir / pair_name / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    tag = f"src{args.src_prior}_tgt{args.tgt_prior}"
    stats_path   = seed_dir / f"{args.arch}_stats_{tag}.txt"
    args_path    = seed_dir / f"{args.arch}_args_{tag}.json"
    ckpt_path    = seed_dir / f"{args.arch}_checkpoint_{tag}.pth"
    encoder_path = seed_dir / f"{args.arch}_{tag}.pth"

    # 기록 파일 열기 (기존 stats.txt → stats_{tag}.txt)
    stats_file = open(stats_path, 'a', buffering=1)
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    data_manager = DataManager(
        train_dataset = args.src_dataset,
        test_dataset = args.tgt_dataset,
        src_prior = args.src_prior,
        tgt_prior = args.tgt_prior
        )
    
    src_dataset, tgt_dataset = data_manager.get_data(merge=True)
    src_dataset = TwoViewDataset(src_dataset, TransformCL(args.src_dataset))
    tgt_dataset = TwoViewDataset(tgt_dataset, TransformCL(args.tgt_dataset))

    tot_dataset = ConcatDataset([src_dataset, tgt_dataset])
    total_loader = DataLoader(
        tot_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True)


    backbone = build_model(arch=args.arch)
    model = BarlowTwins(
        args,
        backbone=backbone,
       ).cuda(args.gpu_id)
    

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)


    # automatically resume from checkpoint if it exists
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    start_time = time.time()
    scaler = GradScaler()
    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        for step, ((y1, y2), _, _) in enumerate(total_loader, start=epoch * len(total_loader)):
            y1 = y1.cuda(args.gpu_id, non_blocking=True)
            y2 = y2.cuda(args.gpu_id, non_blocking=True)
            adjust_learning_rate(args, optimizer, total_loader, step)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict()
                )
            torch.save(state, ckpt_path)
    if args.rank == 0:
        torch.save(model.backbone.state_dict(), encoder_path)


    
        


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone
        
        # self.backbone.fc = nn.Identity()

        # projector
        feat_dim = self.backbone.feature_dim
        sizes = [feat_dim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2): # len(sizes)-2=2
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        y1 = self.backbone.forward_features(y1)
        y2 = self.backbone.forward_features(y2)
        z1 = self.projector(y1)
        z2 = self.projector(y2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



if __name__ == '__main__':
    main()