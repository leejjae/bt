import warnings
warnings.filterwarnings("ignore")

import copy
import argparse
import pickle
import json
import transformers
import mlflow
import logging
from itertools import cycle
from typing import Optional, Iterator, Iterable

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

# ===== Imports (synthetic data + PUCL + sklearn metrics) =====
import lossFunc
from utils import *
from utils_fixmatch import *

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # metrics via sklearn
from syn_dataset import make_syn_loaders                              # synthetic PU data
from syn_model import PUCL, SwAV, BarlowTwins                                         
from syn_repr_loader import make_syn_representation                   # representation maker

parser = argparse.ArgumentParser(description='PyTorch Implementation of CL+PU')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--ed', default='')

parser.add_argument('--train_dataset', type=str, default='syn')
parser.add_argument('--test_dataset', type=str, default='syn')                 

parser.add_argument('--seed', type=int, default=43, help='random seed (default: 0)')

# GPU
parser.add_argument('--no_cuda', action='store_true',
                    help='disable cuda (default: False)')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='set gpu id to use (default: 0)')

# Pre-training
parser.add_argument('--pre_optim', type=str, default='adam', choices=['adam', 'sgd'],
                    help='name of optimizer for pre-training (default adam)')
parser.add_argument('--pre_epochs', type=int, default=50,
                    help='number of pre-training epochs (default 400)')
parser.add_argument('--pre_lr', type=float, default=1e-3, help='pre-training learning rate (default 1e-3)')
parser.add_argument('--pre_wd', type=float, default=0., help='weight decay for pre-training (default 0.)')
parser.add_argument('--pre_batch_size', type=int, default=512, help='batch size for pre-training (default 64)')
parser.add_argument('--pre_n_warmup', type=int, default=0,
                    help='number of warm-up steps in pre-training (default 0)')
parser.add_argument('--pre_cos', action='store_true',
                    help='Use cosine lr scheduler in pre-training (default False)')
parser.add_argument('--pretrained', type=str,
                    default=None,
                    help='./models/cifar_0.5_0.001_400_nnpu_64.pth')

# Training
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs to run (default: 100)')
parser.add_argument('--batch_size', default=510, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=510, type=int,
                    help='mini-batch size of validation (default: 128)')


parser.add_argument('--optim', type=str, default='adam', help='type of optimizer for training (default: adam)')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--wd', default=0., type=float, help='weight decay (default 0.)')
parser.add_argument('--decay_epoch', default=-1, type=int,
                    help='Reduces the learning rate every decay_epoch (default -1)')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')
parser.add_argument('--cos', action='store_true',
                    help='Use cosine lr scheduler (default False)')
parser.add_argument('--n_warmup', default=0, type=int,
                    help='Number of warm-up steps (default: 0)')
parser.add_argument('--patience', default=5, type=int, help='patience for early stopping (default 5)')
parser.add_argument('--restart', action='store_true',
                    help='reset model before training in each episode (default: False)')

# Test
parser.add_argument('--run_all', action='store_true', help='run all experiences with 20 seeds (default False)')

# CL
parser.add_argument('--inner_epochs', type=int, default=1,
                    help='number of epochs to run after each dataset update (default: 1)')
parser.add_argument('--max_thresh_p', type=float, default=2., help='maximum of threshold for labeled (default 2.0)')
parser.add_argument('--max_thresh_n', type=float, default=2., help='maximum of threshold for unlabeled (default 2.0)')
parser.add_argument('--grow_steps_p', type=int, default=10, help='number of step to grow to max_thresh for labeled (default 10)')
parser.add_argument('--grow_steps_n', type=int, default=10, help='number of step to grow to max_thresh for unlabeled (default 10)')
parser.add_argument('--scheduler_type_p', type=str, default='concave',
                    help='type of training scheduler for labeled (default linear)')
parser.add_argument('--scheduler_type_n', type=str, default='concave',
                    help='type of training scheduler for unlabeled (default linear)')
parser.add_argument('--alpha_p', type=float, default=0.1, help='initial threshold for labeled (default 0.1)')
parser.add_argument('--alpha_n', type=float, default=0.1, help='initial threshold for unlabeled (default 0.1)')
parser.add_argument('--eta', type=float, default=1.1,
                    help='alpha *= eta in each step for scheduler exp (default 1.1)')
parser.add_argument('--p', type=int, default=2, help='p for scheduler root-p (default 2)')
parser.add_argument('--spl_type', type=str, default='welsch', help='type of soft sp-regularizer (default welsch)')
parser.add_argument('--mix2_gamma', type=float, default=1.0, help='gamma in mixture2 (default 1.0)')
parser.add_argument('--poly_t', type=int, default=3, help='t in polynomial (default 3)')

# PU
parser.add_argument('--prior', type=float, default=0.5)
parser.add_argument("--src_prior", default=0.5, type=float)
parser.add_argument("--tgt_prior", default=0.5, type=float)
parser.add_argument('--pre_loss', type=str, default='nnpu', choices=['bce', 'nnpu', 'upu'])
parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'nnpu', 'upu', 'focal'])
parser.add_argument('--focal_gamma', type=float, default=1.0, help='gamma for focal loss')
parser.add_argument('--hardness', type=str, default='logistic',
                        help='hardness function used to calculate weights (default: logistic)')
parser.add_argument('--temper_n', type=float, default=0.5, help='temperature to smooth logits for unlabeled (default: 1.0)')
parser.add_argument('--temper_p', type=float, default=0.5, help='temperature to smooth logits for labeled (default: 1.0)')
parser.add_argument('--phi', type=float, default=0., help='momentum for weight moving average (default: 0.)')
parser.add_argument('--encoder', type=str, default='pucl')
parser.add_argument('--n_prototypes', type=int, default=10)


# FixMatch
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--pos_thresh', default=0.7, type=float,help='pseudo label threshold')
parser.add_argument('--neg_thresh', default=0.3, type=float,help='pseudo label threshold')
parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lam_u', default=1.0, type=float, help='coefficient of unlabeled loss')

parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument("--hidden_mlp", default=128, type=int)
parser.add_argument("--feat_dim", default=128, type=int)    # PUCL repr dim = 128
parser.add_argument("--nmb_prototypes", default=30, type=int)
parser.add_argument("--in_channels", default=3, type=int)
parser.add_argument("--ckpt", default=None, type=str)       # optional
parser.add_argument("--output_dir", default="experiment/fixrobust", type=str)
args = parser.parse_args()
global_step = 0
moving_weights_all = None



def compute_metrics(y_true_np: np.ndarray, y_score_np: np.ndarray):
    # y_true in {-1, +1}, y_score in [0,1] as positive prob.
    y_bin = (y_true_np == 1).astype(int)
    y_pred_bin = (y_score_np >= 0.5).astype(int)
    y_pred_pm = np.where(y_pred_bin == 1, 1, -1)

    acc = accuracy_score(np.where(y_bin == 1, 1, -1), y_pred_pm)
    f1 = f1_score(np.where(y_bin == 1, 1, -1), y_pred_pm, average="macro")
    try:
        auc = roc_auc_score(y_bin, y_score_np)
    except ValueError:
        auc = float("nan")  # single-class edge case
    return acc, f1, auc


def evaluate_model(model, loader, gpu_id):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y_true, _ in loader:
            x = x.cuda()
            y_true = y_true.cuda()
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            ps.append(probs)
            ys.append(y_true.detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0)
    p_np = np.concatenate(ps, axis=0)
    return compute_metrics(y_np, p_np)




# pre_train(classifier, weak_train_z_loader, test_z_loader, args)
def pre_train(classifier, train_loader, val_loader, test_loader, args):
    lr = args.pre_lr
    epochs = args.pre_epochs
    n_warmup = args.pre_n_warmup
    if args.pre_optim == 'adam':
        optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=args.pre_wd)
    elif args.pre_optim == 'sgd':
        optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.8, weight_decay=args.pre_wd)
    else:
        raise ValueError(f'Invalid optimizer name {args.pre_optim}')
    if args.pre_cos:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup,
                                                                 num_training_steps=epochs)
    else:
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup)


    best_val_acc = 0.0
    # best_state = copy.deepcopy(classifier.state_dict())

    for epoch in range(epochs):
        classifier.train()
        for data, true_labels, labels in train_loader:
            data, true_labels, labels = data.cuda(), true_labels.cuda(), labels.cuda()
            true_labels[true_labels==0] = -1
            labels[labels==0] = -1  
            # print(f'labels: {labels}')
            logits = classifier(data).squeeze(-1)
            loss = lossFunc.nnpu_loss(logits, labels, prior=args.src_prior)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc, val_f1, val_auc = evaluate_model(classifier, val_loader, args.gpu_id )
        test_acc, test_f1, test_auc = evaluate_model(classifier, test_loader, args.gpu_id)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(classifier.state_dict())

        print(f"[pretrain {epoch+1:03d}/{epochs}] "
              f"test_acc={test_acc:.4f} test_f1={test_f1:.4f} test_auc={test_auc:.4f}")
    
    # classifier.load_state_dict(best_state)
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(classifier.state_dict(),
               f'models/{args.test_dataset}_{args.src_prior}_{lr}_{epochs}_{args.pre_loss}_{args.pre_batch_size}.pth')








def weighted_dataloader(classifier, dataloader, thresh_p, thresh_n, args):
    classifier.eval()
    data_all, labels_all, true_labels_all, weights_all, probs_all, fea_all = [], [], [], [], [], []
    global moving_weights_all
    with torch.no_grad():
        for data, true_labels, labels in dataloader:
            if args.cuda:
                data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()

            net_out = classifier(data)

            data_all.append(data)
            labels_all.append(labels)
            true_labels_all.append(true_labels)

            probs = torch.sigmoid(net_out)
            probs_all.append(probs)

            if args.hardness in ['distance', 'cos']:
                continue

            if args.hardness == 'logistic':
                loss = lossFunc.logistic_loss(net_out / args.temper_n, -1)
            elif args.hardness == 'sigmoid':
                loss = lossFunc.sigmoid_loss(net_out / args.temper_n, -1)
            elif args.hardness == 'crps':
                loss = lossFunc.crps(net_out / args.temper_n, -1)
            elif args.hardness == 'brier':
                loss = lossFunc.brier(net_out / args.temper_n, -1)
            elif args.hardness == 'focal':
                loss = lossFunc.b_focal_loss(net_out / args.temper_n, -1 * torch.ones_like(labels), gamma=args.focal_gamma, reduction='none')
            else:
                raise ValueError(f'Invalid surrogate loss function {args.hardness}')
            weights = calculate_spl_weights(loss.detach(), thresh_n, args)

            if args.hardness == 'logistic':
                loss = lossFunc.logistic_loss(net_out / args.temper_p, 1)
            elif args.hardness == 'sigmoid':
                loss = lossFunc.sigmoid_loss(net_out / args.temper_p, 1)
            elif args.hardness == 'crps':
                loss = lossFunc.crps(net_out / args.temper_p, 1)
            elif args.hardness == 'brier':
                loss = lossFunc.brier(net_out / args.temper_p, 1)
            elif args.hardness == 'focal':
                loss = lossFunc.b_focal_loss(net_out / args.temper_p, torch.ones_like(labels), gamma=args.focal_gamma, reduction='none')
            else:
                raise ValueError(f'Invalid hardness function {args.hardness}')
            weights[labels == 1] = calculate_spl_weights(loss[labels == 1].detach(), thresh_p, args)
            weights_all.append(weights)

        data_all = torch.cat(data_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        true_labels_all = torch.cat(true_labels_all, dim=0)
        weights_all = torch.cat(weights_all, dim=0)
        if moving_weights_all is None:
            moving_weights_all = weights_all
        else:
            moving_weights_all = args.phi * moving_weights_all + (1. - args.phi) * weights_all
        probs_all = torch.cat(probs_all, dim=0)

    dataloader = DataLoader(TensorDataset(data_all, true_labels_all, labels_all, moving_weights_all), shuffle=True,
                            batch_size=args.batch_size, drop_last=True)
    return dataloader



def train_episode(classifier, weighted_train_dataloader, test_loader, args):
    if args.optim == 'adam':
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    if args.cos:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.n_warmup,
                                                                 num_training_steps=args.inner_epochs)
    else:
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.n_warmup)

    classifier.eval()
    with torch.no_grad():
        meter = AverageMeter()
        for data, true_labels, labels, weights in weighted_train_dataloader:
            if args.cuda:
                data, true_labels, labels, weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()

            net_out = classifier(data).squeeze(1)

            if args.loss == 'bce':
                loss = lossFunc.bce_loss(net_out, labels, weights)
            elif args.loss == 'focal':
                loss = lossFunc.b_focal_loss(net_out, labels, weights, gamma=args.focal_gamma)
            else:
                loss = getattr(lossFunc, f'{args.loss}_loss')(net_out, labels, args.src_prior, weights)
            meter.update(loss.item(), labels.size(0))

    if args.restart:
        classifier.reset_para()

    classifier.train()
    

    tot_loss_meter = AverageMeter()
    tot_true_loss_meter = AverageMeter()
    tot_acc_meter = AverageMeter()

    # from tqdm import tqdm
    # def infinite(loader: Iterable) -> Iterator:
    #     while True:
    #         for batch in tqdm(loader):
    #             yield batch
    # test_iter = infinite(test_loader)
    # from itertools import chain, repeat
    # test_iter = iter(chain.from_iterable(repeat(test_loader)))

    test_base_loader = test_loader          # ← 반드시 torch.utils.data.DataLoader 여야 함
    test_iter = iter(test_base_loader)

    for inner_epoch in range(args.epochs):
        loss_meter = AverageMeter()
        for data, labels, true_labels, weights in weighted_train_dataloader:
            try:
                test_data, test_labels, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_base_loader)    # iterator가 아니라 DataLoader에서 재생성
                test_data, test_labels, _ = next(test_iter)

            if args.cuda:
                data, true_labels, labels,  weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()
                test_data, test_labels = test_data.cuda(non_blocking=True), test_labels.cuda(non_blocking=True)
            
            B = data.size(0)
            MuB = args.mu * B

            u = test_data
            n = u.size(0)
            if n != MuB:
                k = (MuB + n - 1) // n
                shape = (k,) + (1,) * (u.dim() - 1)
                u = u.repeat(shape)[:MuB]


            # weak/strong 증강 없이 "원본 그대로" 두 갈래로 사용
            weak_test_z  = u
            strong_test_z = u.clone()

            # 아래 4줄은 기존 FixMatch 핵심 흐름 그대로 유지
            batch_ = torch.cat([data, weak_test_z, strong_test_z], dim=0)
            batch  = interleave(batch_, 2 * args.mu + 1)

            logits = classifier(batch)
            logits = de_interleave(logits, 2 * args.mu + 1)

            logits_x = logits[:B]
            logits_u_w, logits_u_s = logits[B:].chunk(2)  # weak for pseudo-label, strong for consistency
            del logits

            labels = labels.unsqueeze(-1)
            true_labels = true_labels.unsqueeze(-1)

            if args.loss == 'bce':
                loss_src = lossFunc.bce_loss(logits_x, labels, weights)
            elif args.loss == 'focal':
                loss_src = lossFunc.b_focal_loss(logits_x, labels, weights, gamma=args.focal_gamma)
            else:
                loss_src = getattr(lossFunc, f'{args.loss}_loss')(logits_x, labels, args.src_prior, weights)

            # target, mask_pos, mask_neg, mask = make_labels(logits_u_w.squeeze(1), 0.95, 0.05, hard=True)
            target, mask = make_labels_from_fixmatch(logits_u_w, tau=0.98, T=1)
            loss_tgt = F.binary_cross_entropy_with_logits(logits_u_s, target, reduction='none')
            loss_tgt = (loss_tgt * mask).mean() 
            
            # loss_tgt = lossFunc.bce_loss(logits_u_s.squeeze(1), target, mask)


            loss = loss_src + args.lam_u * loss_tgt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            true_loss = lossFunc.bce_loss(logits_x, true_labels, weights)
            acc = accuracy(logits_x, true_labels)

            tot_loss_meter.update(loss.item(), data.size(0))
            tot_acc_meter.update(acc, data.size(0))
            tot_true_loss_meter.update(true_loss.item(), data.size(0))
            loss_meter.update(loss.item(), labels.size(0))

        scheduler.step()

        eval_acc, eval_f1, eval_auc = evaluate_model(classifier, test_loader, args.gpu_id)
        
      
    return tot_loss_meter.avg, tot_acc_meter.avg, tot_true_loss_meter.avg



def train(classifier, positive_dataset, unlabeled_dataset, val_dataset, test_dataset, args):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False ,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False ,drop_last=True)
    # tot_test_loader = DataLoader(tot_test_dataset, batch_size=args.batch_size_val*args.mu, shuffle=False ,drop_last=True)

    # val_loader  = val_dataset  if isinstance(val_dataset,  DataLoader) else \
    #              DataLoader(val_dataset,  batch_size=args.batch_size_val, shuffle=False, drop_last=False)
    
    # test_loader = test_dataset if isinstance(test_dataset, DataLoader) else \
    #              DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=False)

    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience

    positive_data = positive_dataset.tensors[0]
    positive_labels = positive_dataset.tensors[1]

    unlabeled_data = unlabeled_dataset.tensors[0]
    unl_label = unlabeled_dataset.tensors[2] 
    unlabeled_dataset.tensors[2][unl_label==0]=-1
    unlabeled_labels = unlabeled_dataset.tensors[2]

    cl_scheduler_p = TrainingScheduler(args.scheduler_type_p, args.alpha_p, args.max_thresh_p, args.grow_steps_p, args.p,
                                       args.eta)
    cl_scheduler_n = TrainingScheduler(args.scheduler_type_n, args.alpha_n, args.max_thresh_n, args.grow_steps_n, args.p,
                                       args.eta)

    history_loss = []
    history_acc = []
    history_true_loss = []
    history_val_loss = []
    history_val_acc = []

    val_best_acc = 0.
    val_best_index = -1
    val_best_model = copy.deepcopy(classifier.state_dict())

    fea_all = []
    history_test_acc = []
    history_test_macro_f1 = []  
    history_test_auc = []
    for episode in range(epochs): 
        thresh_p = cl_scheduler_p.get_next_ratio()
        thresh_n = cl_scheduler_n.get_next_ratio()

        cur_data = torch.cat((positive_data, unlabeled_data), dim=0)
        cur_labels = torch.cat((positive_labels, -torch.ones_like(unlabeled_labels)), dim=0)
        cur_true_labels = torch.cat((positive_labels, unlabeled_labels), dim=0)
        perm = np.random.permutation(cur_data.size(0))
        cur_data, cur_labels, cur_true_labels = cur_data[perm], cur_labels[perm], cur_true_labels[perm]
        cur_loader = DataLoader(TensorDataset(cur_data,cur_true_labels, cur_labels), batch_size=batch_size,
                                shuffle=True)
        weighted_train_loader = weighted_dataloader(classifier, cur_loader, thresh_p, thresh_n, args)

        tot_loss, tot_acc, tot_true_loss = train_episode(classifier, weighted_train_loader, test_loader, args)

        val_acc, val_macro_f1, val_auc = evaluate_model(classifier, val_loader, args)
        test_acc, test_macro_f1, test_auc = evaluate_model(classifier, test_loader, args)

        print(f"[train_episode {episode+1:03d}/{epochs}] "
              f"test_acc={test_acc:.4f} test_f1={test_macro_f1:.4f} test_auc={test_auc:.4f}")


        history_loss.append(tot_loss)
        history_acc.append(tot_acc)
        history_true_loss.append(tot_true_loss)
        # history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        history_test_acc.append(test_acc)
        history_test_macro_f1.append(test_macro_f1)
        history_test_auc.append(test_auc)

        # if args.debug:
        #     mlflow.log_metric('val_loss', val_loss)
        #     mlflow.log_metric('val_err', 100.0 - val_acc * 100.0)

        if val_acc > val_best_acc:
            val_best_acc = val_acc
            val_best_index = episode
            val_best_model = copy.deepcopy(classifier.state_dict())

    classifier.load_state_dict(val_best_model)

    history = {'pseudo_loss': history_loss, 'true_loss': history_true_loss, 'acc': history_acc,
               'val_loss': history_val_loss, 'val_acc': history_val_acc,
               'test_acc' : history_test_acc, 'test_macro_f1': history_test_macro_f1, 'test_auc': history_test_auc}

    return history

def prepare_and_run(args):
    seed_all(args.seed)


    if args.encoder == "pucl":
        model = PUCL().cuda().eval()
    elif args.encoder == "swav":
        model = SwAV(n_prototypes=args.n_prototypes).cuda().eval()
    elif args.encoder == "barlow":
        model = BarlowTwins().cuda().eval()


    state_dict = torch.load(args.ckpt, map_location="cpu")
    state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    

    # Synthetic PU datasets
    src_dataset,tgt_dataset = make_syn_loaders(
        n_src_pos=1000,
        n_src_u=5000,
        n_tgt_u=5000,
        src_prior=args.src_prior,
        tgt_prior=args.tgt_prior,
        d_in=5,
        d_sp=20,
        gamma=0.5,
        sigma_in2=0.05,
        sigma_sp2=1.0,
        seed=args.seed,
        noise_std=1e-3,
        same_noise=False,
    )


    # make val_dataset
    n_total = len(src_dataset)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    
    src_dataset, val_dataset = random_split(
        src_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
        )
    

    print(f"len(src_dataset): {len(src_dataset)}")
    print(f"len(val_dataset): {len(val_dataset)}")


    src_x_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # val_x_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # tgt_x_loader = DataLoader(tgt_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

    val_x_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    tgt_x_loader = DataLoader(tgt_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

    
    src_z_loader, src_p_dataset, src_u_dataset = make_syn_representation(
        model.backbone,
        src_x_loader,
        args.gpu_id,
        train=True,
        batch_size=510)


    val_z_loader, val_z_dataset = make_syn_representation(
        model.backbone,
        val_x_loader,
        args.gpu_id,
        train=False,
        batch_size=510)
    

    tgt_z_loader, tgt_z_dataset = make_syn_representation(
        model.backbone,
        tgt_x_loader,
        args.gpu_id,
        train=False,
        batch_size=510)
    # print(f'len(tgt_z_loader): {len(tgt_z_loader)}')
    
    classifier = model.classifier.cuda()



    if args.pretrained:
        print(f'Model loaded from: {args.pretrained}.')
        classifier.load_state_dict(torch.load(args.pretrained))
    else:
        print(f'len(src_z_loader): {len(src_z_loader)}')
        print(f'len(val_z_loader): {len(val_z_loader)}')
        pre_train(classifier, src_z_loader, val_z_loader, tgt_z_loader, args)

    

    test_acc, test_macro_f1, test_auc = evaluate_model(classifier, tgt_z_loader, args)

    history = train(classifier, src_p_dataset, src_u_dataset, val_z_dataset, tgt_z_dataset, args)

    test_acc, test_macro_1, test_auc2 = evaluate_model(classifier, tgt_z_loader, args)
    test_err = 1. - test_acc

    return history

def main():
    args = parser.parse_args()
    seed_all(args.seed)
    args.cuda = (not args.no_cuda)
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
    if args.run_all:
        args.debug = False

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    history = prepare_and_run(args)
    epochs = range(1, len(history['acc']) + 1)

    max_test_acc = float(np.max(history['test_acc']))
    max_test_macro_f1  = float(np.max(history['test_macro_f1']))
    max_test_auc = float(np.max(history.get('test_auc', [float('nan')])))

    plt.figure(figsize=(10,6))
    plt.plot(epochs, history['acc'],             label='Train Acc')
    plt.plot(epochs, history['test_acc'],        label='Test Acc')
    plt.plot(epochs, history['test_macro_f1'],   label='Test macro F1-score')
    if 'test_auc' in history:
        plt.plot(epochs, history['test_auc'],    label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(f"seed_{args.seed}/{args.test_dataset}/acc:{max_test_acc:.4f}/f1:{max_test_macro_f1:.4f}/auc:{max_test_auc:.4f}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_name = f"{args.seed}_{args.src_prior}_{args.tgt_prior}_{args.test_dataset}.png"
    plot_path = os.path.join(args.output_dir, plot_name)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✅  Accuracy curve saved to {plot_path}")

    # Per instruction: remove txt aggregation from main.py
    seed_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)

    with open(os.path.join(seed_dir, "history.json"), "w") as f:
        json.dump({
            "acc": history["test_acc"],
            "macro_f1": history["test_macro_f1"],
            "auc": history.get("test_auc", []),
        }, f, indent=2)

if __name__ == '__main__':
    main()
