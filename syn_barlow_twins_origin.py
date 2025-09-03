import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch

from syn_dataset import make_syn_loaders
from syn_model import SynEncoder

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', nargs='?', default='./temp_path', help='path to dataset')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning_rate_weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning_rate_biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight_decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print_freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--gpu_id', default=0, type=int)

parser.add_argument('--src_prior', default=0.5, type=float)
parser.add_argument('--tgt_prior', default=0.5, type=float) 

parser.add_argument('--hidden_mlp', default=64, type=int)
parser.add_argument('--feat_dim', default=32, type=int)


def main():
    args = parser.parse_args()
    args.rank = 0
    
    
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    _, _, tot_dataset = make_syn_loaders(
        n_src_pos=1000,
        n_src_u=5000,
        n_tgt_u=5000,
        src_prior=args.src_prior,
        tgt_prior=args.tgt_prior,
        d_in=5, d_sp=20,
        gamma=0.5,
        sigma_in2=0.05,
        sigma_sp2=1.0,
        seed=42
    )
    total_loader = DataLoader(tot_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    batch = next(iter(total_loader))[0]   # inputs
    input_dim = batch[0].shape[-1]
    del batch

    syn_backbone = SynEncoder(input_dim=input_dim, hidden_mlp=args.hidden_mlp, feat_dim=args.feat_dim)
    
    model = BarlowTwins(
        args,
        backbone=syn_backbone,
        feat_dim=args.feat_dim).cuda(args.gpu_id)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_id])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
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
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.backbone.state_dict(),
           args.checkpoint_dir / 'syn.pth')


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


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module): 
    def __init__(self, args, backbone, feat_dim):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.backbone.fc = nn.Identity()


        # projector
        sizes = [feat_dim] + [feat_dim*4]*3
        layers = []
        for i in range(len(sizes) - 2):         # i=0,1
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=False))  
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False)) 
        self.projector = nn.Sequential(*layers)
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

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