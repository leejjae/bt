import warnings
warnings.filterwarnings("ignore")

import copy
import argparse
import pickle
import transformers
import mlflow
import logging
from itertools import cycle

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

# import helpers
# import models
import lossFunc
import src.resnet_50
from utils import * 
import dataset
from repr_loader import *
from fixmatch_utils import *

parser = argparse.ArgumentParser(description='PyTorch Implementation of CL+PU')


parser.add_argument('--debug', action='store_true')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--ed', default='')
# Dirs
# parser.add_argument("--output_dir", type=str, default=os.getenv("AMLT_OUTPUT_DIR", "results"),
#                     help='output dir (default: results)')
# parser.add_argument("--data_dir", type=str, default=os.getenv("AMLT_DATA_DIR", "data"),
#                         help="Directory where dataset is stored")
# # Dataset

parser.add_argument('--train_dataset', type=str, default='cifar')
parser.add_argument('--test_dataset', type=str, default='cifar10c')                 


parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')


# GPU
parser.add_argument('--no_cuda', action='store_true',
                    help='disable cuda (default: False)')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='set gpu id to use (default: 0)')

# Pre-training
parser.add_argument('--pre_optim', type=str, default='adam', choices=['adam', 'sgd'],
                    help='name of optimizer for pre-training (default adam)')
parser.add_argument('--pre_epochs', type=int, default=200,
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

# './models/cinic_0.5_0.001_100_nnpu_64.pth'
# Training
parser.add_argument('--epochs', type=int, default=300,
                    help='number of training epochs to run (default: 100)')
parser.add_argument('--batch_size', default=512, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=512, type=int,
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
parser.add_argument("--src_prior", default=0.3, type=float)
parser.add_argument("--tgt_prior", default=0.7, type=float)
parser.add_argument('--pre_loss', type=str, default='nnpu', choices=['bce', 'nnpu', 'upu'])
parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'nnpu', 'upu', 'focal'])
parser.add_argument('--focal_gamma', type=float, default=1.0, help='gamma for focal loss')
parser.add_argument('--hardness', type=str, default='logistic',
                        help='hardness function used to calculate weights (default: logistic)')
parser.add_argument('--temper_n', type=float, default=0.5, help='temperature to smooth logits for unlabeled (default: 1.0)')
parser.add_argument('--temper_p', type=float, default=0.5, help='temperature to smooth logits for labeled (default: 1.0)')
parser.add_argument('--phi', type=float, default=0., help='momentum for weight moving average (default: 0.)')

# FixMatch
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--pos_thresh', default=0.7, type=float,help='pseudo label threshold')
parser.add_argument('--neg_thresh', default=0.3, type=float,help='pseudo label threshold')
parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lam_u', default=1.0, type=float, help='coefficient of unlabeled loss')

parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument("--hidden_mlp", default=128, type=int)
parser.add_argument("--feat_dim", default=64, type=int)
parser.add_argument("--nmb_prototypes", default=30, type=int)
parser.add_argument("--in_channels", default=3, type=int)
parser.add_argument("--ckpt", required=True, type=str)
parser.add_argument("--output_dir", default="experiment/fixrobust", type=str)
args = parser.parse_args()
global_step = 0
moving_weights_all = None


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(args.feat_dim,32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,1)
        self.fc3 = nn.Linear(args.feat_dim,1)
    def forward(self, x):
        if args.feat_dim ==128:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
        else:
            return self.fc3(x)
    
def batch_confusion_counts(logits: torch.Tensor, y_true: torch.Tensor):
    """
    logits: shape (B,) or (B,1) – raw scores (threshold 0)
    y_true: in {-1, +1}; we map to {0,1} inside
    Returns: tp, fp, fn, tn (ints)
    """
    if logits.dim() > 1:
        logits = logits.squeeze(1)
    preds   = (logits >= 0).long()
    targets = (y_true.clone() == 1).long()   # {1,-1} -> {1,0}

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    return tp, fp, fn, tn


def macro_f1_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-8) -> float:
    """
    Macro-F1 for binary case = (F1_pos + F1_neg) / 2
    F1_pos = 2*TP / (2*TP + FP + FN)
    F1_neg = 2*TN / (2*TN + FN + FP)
    """
    f1_pos = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    f1_neg = (2.0 * tn) / (2.0 * tn + fn + fp + eps)
    return 0.5 * (f1_pos + f1_neg)


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

    history_loss = []
    history_acc = []

    best_val_acc = 0.0
    best_val_classifier = copy.deepcopy(classifier.state_dict())
    best_epoch=-1
    patience=10
    min_es_epoch = 20 
    for epoch in range(epochs):
        loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        classifier.train()
        for data, true_labels, labels in train_loader:
            data, true_labels, labels = data.cuda(), true_labels.cuda(), labels.cuda()
            true_labels[true_labels==0] = -1 ## PU learning에서는 neg label : -1
            net_out= classifier(data).squeeze(1)  

            loss = lossFunc.nnpu_loss(net_out, labels, prior=args.src_prior)
            train_acc = accuracy(net_out, true_labels)

            loss_meter.update(loss.item(), data.size(0))
            train_acc_meter.update(train_acc, data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        val_loss, val_acc, val_macro_f1 = test(classifier, val_loader, args)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_classifier = copy.deepcopy(classifier.state_dict())
            best_epoch = epoch

        scheduler.step()
        test_loss, test_acc, test_macro_f1 = test(classifier, test_loader, args)

        history_loss.append(loss_meter.avg)
        history_acc.append(train_acc_meter.avg)

      
    history = {'loss': history_loss, 'acc': history_acc}
    classifier.load_state_dict(best_val_classifier)
    
    

    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(classifier.state_dict(),
               f'models/{args.test_dataset}_{args.src_prior}_{lr}_{epochs}_{args.pre_loss}_{args.pre_batch_size}.pth')
    return history

# test_loss, test_acc = test(classifier, test_loader, args)
def test(classifier, loader, args):
    classifier.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    TP = FP = FN = TN = 0
    with torch.no_grad():
        for data, true_labels, labels  in loader:
            data, true_labels, labels = data.cuda(), true_labels.cuda(), labels.cuda()

            true_labels[true_labels==0] = -1
            
            net_out = classifier(data).squeeze(1)
            loss = lossFunc.bce_loss(net_out, labels)
            acc = accuracy(net_out, true_labels)

            loss_meter.update(loss.item(), data.size(0))
            acc_meter.update(acc, data.size(0))

            tp, fp, fn, tn = batch_confusion_counts(net_out, true_labels)
            TP += tp; FP += fp; FN += fn; TN += tn
    macro_f1 = macro_f1_from_counts(TP, FP, FN, TN)
    return loss_meter.avg, acc_meter.avg, macro_f1



# weighted_train_loader = weighted_dataloader(classifier, cur_loader, thresh_p, thresh_n, args)
def weighted_dataloader(classifier, dataloader, thresh_p, thresh_n, args):
    # calculate weights for all
    # training = model.training
    classifier.eval()
    data_all, labels_all, true_labels_all, weights_all, probs_all, fea_all = [], [], [], [], [], []
    global moving_weights_all
    with torch.no_grad():
        # dataloader : weak_
        for data, true_labels, labels in dataloader:
            if args.cuda:
                data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()

            if args.hardness in ['distance', 'cos']:
                net_out = classifier(data, return_fea=True)
                # fea_all.append(fea)
            else:
                net_out = classifier(data)

            data_all.append(data)
            labels_all.append(labels)
            true_labels_all.append(true_labels)

            # unlabeled data with linear weight
            probs = torch.sigmoid(net_out)
            probs_all.append(probs)

            if args.hardness in ['distance', 'cos']:
                continue

            # loss for calculating unlabeled weight
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
            # weights for unlabeled
            weights = calculate_spl_weights(loss.detach(), thresh_n, args)

            # loss for calculating labeled weight
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
            # weights for labeled
            weights[labels == 1] = calculate_spl_weights(loss[labels == 1].detach(), thresh_p, args)
            weights_all.append(weights)

        data_all = torch.cat(data_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        true_labels_all = torch.cat(true_labels_all, dim=0)
        if args.hardness == 'distance':
            fea_all = torch.cat(fea_all, dim=0)
            p_fea = fea_all[labels_all == 1]
            u_fea = fea_all[labels_all == -1]
            p_mean = p_fea.mean(dim=0)
            u_mean = u_fea.mean(dim=0)
            p_dis = euclidean_distance(fea_all, p_mean)
            u_dis = euclidean_distance(fea_all, u_mean)
            weights_all = torch.where(labels_all == 1, calculate_spl_weights(p_dis / u_dis, thresh_p, args), calculate_spl_weights(u_dis / p_dis, thresh_n, args))
        elif args.hardness == 'cos':
            fea_all = torch.cat(fea_all, dim=0)
            p_fea = fea_all[labels_all == 1]
            u_fea = fea_all[labels_all == -1]
            p_mean = p_fea.mean(dim=0)
            u_mean = u_fea.mean(dim=0)
            p_sim = F.cosine_similarity(fea_all, p_mean)
            u_sim = F.cosine_similarity(fea_all, u_mean)
            weights_all = torch.where(labels_all == 1, calculate_spl_weights(1. - p_sim, thresh_p, args), calculate_spl_weights(1 - u_sim, thresh_n, args))
        else:
            weights_all = torch.cat(weights_all, dim=0)
        if moving_weights_all is None:
            moving_weights_all = weights_all
        else:
            moving_weights_all = args.phi * moving_weights_all + (1. - args.phi) * weights_all
        probs_all = torch.cat(probs_all, dim=0)

        unlabel_weights = moving_weights_all[labels_all == -1]
        unlabel_true_labels = true_labels_all[labels_all == -1]
        unlabel_probs = probs_all[labels_all == -1]
        

    dataloader = DataLoader(TensorDataset(data_all, true_labels_all, labels_all, moving_weights_all), shuffle=True,
                            batch_size=args.batch_size, drop_last=True)
    # model.train(training)
    return dataloader


# train_episode(classifier, weighted_train_loader, tot_test_loader, args)
def train_episode(classifier, weighted_train_dataloader, test_loader, args):
    #####################################
    ###### FixMatch Algorithm 추가 ######
    #####################################
    if args.optim == 'adam':
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)


    if args.cos:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.n_warmup,
                                                                 num_training_steps=args.inner_epochs)
    else:
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.n_warmup)

    # Test on training set before training
    classifier.eval()
    with torch.no_grad():
        meter = AverageMeter()
        for data, true_labels, labels, weights in weighted_train_dataloader:
            if args.cuda:
                data, true_labels, labels, weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()

            net_out = classifier(data).squeeze(1)
            # loss w.r.t. pseudo labels
            if args.loss == 'bce':
                loss = lossFunc.bce_loss(net_out, labels, weights)
            elif args.loss == 'focal':
                loss = lossFunc.b_focal_loss(net_out, labels, weights, gamma=args.focal_gamma)
            else:
                loss = getattr(lossFunc, f'{args.loss}_loss')(net_out, labels, args.src_prior, weights)
            meter.update(loss.item(), labels.size(0))
        # logging.info(f'Loss before training: {meter.avg}')

    if args.restart:
        classifier.reset_para()

    classifier.train()
    test_iter = cycle(test_loader)

    tot_loss_meter = AverageMeter()
    tot_true_loss_meter = AverageMeter()
    tot_acc_meter = AverageMeter()

    
    for inner_epoch in range(args.inner_epochs):
        loss_meter = AverageMeter()
        pseudo_pos_cnt = 0       # Np‑pred
        pseudo_neg_cnt = 0       # Nn‑pred
        correct_pos_cnt = 0      # TP  (= pseudo==1 & y==1)
        correct_neg_cnt = 0
        for data, labels, true_labels, weights in weighted_train_dataloader:
            (weak_test_z, strong_test_z), _, _ = next(test_iter)
            if args.cuda:
                data, true_labels, labels,  weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()
                weak_test_z, strong_test_z = weak_test_z.cuda(non_blocking=True), strong_test_z.cuda(non_blocking=True)
            
            batch_ = torch.cat([data, weak_test_z, strong_test_z])
            batch = interleave(batch_, 2*args.mu + 1)

            logits = classifier(batch)
            logits = de_interleave(logits, 2*args.mu + 1)

            B = data.size(0)
            logits_x = logits[:B]
            logits_u_w, logits_u_s = logits[B:].chunk(2)
            del logits

            labels = labels.unsqueeze(-1)
            true_labels = true_labels.unsqueeze(-1)

            
            # loss w.r.t. pseudo labels
            if args.loss == 'bce':
                loss_src = lossFunc.bce_loss(logits_x, labels, weights)
                # if args.debug:
                #     net_out_u = net_out[labels == -1]
                #     true_labels_u = true_labels[labels == -1]
                #     loss_up = getattr(lossFunc, f'{args.hardness}_loss')(net_out_u[true_labels_u == 1] / args.temper_n, -1)
                #     loss_un = getattr(lossFunc, f'{args.hardness}_loss')(net_out_u[true_labels_u == -1] / args.temper_n, -1)
                #     loss_p = getattr(lossFunc, f'{args.hardness}_loss')(net_out[labels == 1] / args.temper_p, 1)
                #     global global_step
                #     mlflow.log_metric('loss_up', loss_up.mean().detach().cpu().numpy(), global_step)
                #     mlflow.log_metric('loss_un', loss_un.mean().detach().cpu().numpy(), global_step)
                #     mlflow.log_metric('loss_p', loss_p.mean().detach().cpu().numpy(), global_step)
                #     global_step += 1
            elif args.loss == 'focal':
                loss_src = lossFunc.b_focal_loss(logits_x, labels, weights, gamma=args.focal_gamma)
            else:
                loss_src = getattr(lossFunc, f'{args.loss}_loss')(logits_x, labels, args.src_prior, weights)


            target_prob, mask_pos, mask_neg, mask = make_soft_labels(logits_u_w.squeeze(1),
                                             0.8, 0.2)
        
            

            loss_tgt = lossFunc.bce_loss(logits_u_s.squeeze(1), target_prob, mask)
            loss = loss_src + args.lam_u * loss_tgt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss w.r.t. true labels
            true_loss = lossFunc.bce_loss(logits_x, true_labels, weights)
            # acc w.r.t. true labels
            acc = accuracy(logits_x, true_labels)

            tot_loss_meter.update(loss.item(), data.size(0))
            tot_acc_meter.update(acc, data.size(0))
            tot_true_loss_meter.update(true_loss.item(), data.size(0))

    
            loss_meter.update(loss.item(), labels.size(0))

        scheduler.step()
        if pseudo_pos_cnt > 0:
            acc_pos = correct_pos_cnt / pseudo_pos_cnt
        else:
            acc_pos = float('nan')   # pos 예측이 아예 없으면 표시만 nan

        if pseudo_neg_cnt > 0:
            acc_neg = correct_neg_cnt / pseudo_neg_cnt
        else:
            acc_neg = float('nan')

        # print(f"[Pseudo‑label stats]  "
        #     f"pos‑pred: {pseudo_pos_cnt:6d}  "
        #     f"neg‑pred: {pseudo_neg_cnt:6d}  "
        #     f"pos‑prec: {acc_pos:6.4f}  "
        #     f"neg‑prec: {acc_neg:6.4f}")

        
        # logging.debug(f'inner epoch [{inner_epoch + 1} / {args.inner_epochs}]  train loss: {loss_meter.avg}')
        # if args.debug:
        #     global global_step
        #     mlflow.log_metric('train_loss', loss_meter.avg, global_step)
        #     global_step += 1

    return tot_loss_meter.avg, tot_acc_meter.avg, tot_true_loss_meter.avg

# history = train(
# classifier, pos_z_dataset, unl_z_dataset, weak_val_z_dataset, tot_test_z_dataset, test_z_loader, args)

def train(classifier, positive_dataset, unlabeled_dataset, val_dataset, tot_test_dataset, test_dataset, args):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False ,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False ,drop_last=True)
    tot_test_loader = DataLoader(tot_test_dataset, batch_size=args.batch_size_val*args.mu, shuffle=False ,drop_last=True)

    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience

    ## pos : 1 / neg : -1 / unl : -1 : unlabeled sample를 negative로 간주 ##
    # unlabeled_dataset.dataset.tensors[1][:]=-1

    positive_data = positive_dataset.tensors[0]
    positive_labels = positive_dataset.tensors[1]

    unlabeled_data = unlabeled_dataset.tensors[0]
    unl_label = unlabeled_dataset.tensors[2] 
    unlabeled_dataset.tensors[2][unl_label==0]=-1
    unlabeled_labels = unlabeled_dataset.tensors[2]

    # for SPL
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
    for episode in range(epochs): 
        
        # get next lambda
        thresh_p = cl_scheduler_p.get_next_ratio()
        thresh_n = cl_scheduler_n.get_next_ratio()
        # prRedWhite(f'thresh_p = {thresh_p:.3f}  thresh_n = {thresh_n:.3f}')

        cur_data = torch.cat((positive_data, unlabeled_data), dim=0)
        cur_labels = torch.cat((positive_labels, -torch.ones_like(unlabeled_labels)), dim=0)
        cur_true_labels = torch.cat((positive_labels, unlabeled_labels), dim=0)
        perm = np.random.permutation(cur_data.size(0))
        cur_data, cur_labels, cur_true_labels = cur_data[perm], cur_labels[perm], cur_true_labels[perm]
        cur_loader = DataLoader(TensorDataset(cur_data,cur_true_labels, cur_labels), batch_size=batch_size,
                                shuffle=True)
        weighted_train_loader = weighted_dataloader(classifier, cur_loader, thresh_p, thresh_n, args)

       
        tot_loss, tot_acc, tot_true_loss = train_episode(classifier, weighted_train_loader, tot_test_loader, args)


        val_loss, val_acc, val_macro_f1 = test(classifier, val_loader, args)
        test_loss, test_acc, test_macro_f1 = test(classifier, test_loader, args)
        
        # print(
        #     f'Episode [{episode + 1} / {epochs}]   Pseudo_Loss: {tot_loss:.5f}   test_acc: {test_acc * 100.0:.5f}')

        history_loss.append(tot_loss)
        history_acc.append(tot_acc)
        history_true_loss.append(tot_true_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        history_test_acc.append(test_acc)
        history_test_macro_f1.append(test_macro_f1)
        if args.debug:
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_err', 100.0 - val_acc * 100.0)

        # Early stop
        if val_acc > val_best_acc:
            val_best_acc = val_acc
            val_best_index = episode
            val_best_model = copy.deepcopy(classifier.state_dict())
        else:
            pass
            # if episode - val_best_index >= patience:
            #     print(f'=== Break at epoch {val_best_index + 1} ===')
            #     fea_all = fea_all[:val_best_index + 2]
            #     break

    classifier.load_state_dict(val_best_model)

    
    history = {'pseudo_loss': history_loss, 'true_loss': history_true_loss, 'acc': history_acc,
               'val_loss': history_val_loss, 'val_acc': history_val_acc, 'test_acc' : history_test_acc, 'test_macro_f1': history_test_macro_f1}

    return history


def prepare_and_run(args):
    seed_all(args.seed)

    encoder = src.resnet_50.__dict__[args.arch](
    normalize=True,
    hidden_mlp=args.hidden_mlp,
    output_dim=args.feat_dim,             
    nmb_prototypes=args.nmb_prototypes,
    in_channels=args.in_channels
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval().cuda()


    data_manger = dataset.DataManager(train_dataset=args.train_dataset,
                         test_dataset=args.test_dataset,
                         src_prior=args.src_prior,
                         tgt_prior=args.tgt_prior)
    train_dataset, val_dataset, test_dataset = data_manger.get_data()

    weak_train_x_loader = make_loader(train_dataset,
                                     dataset_name=args.train_dataset,
                                     train=True,
                                     batch_size=args.batch_size)
    
    weak_val_x_loader = make_loader(val_dataset,
                                    dataset_name=args.train_dataset,
                                    train=False,
                                    batch_size=args.batch_size_val)


    test_loader_for_acc = copy.deepcopy(test_dataset)
    test_loader = make_loader(test_loader_for_acc,
                             dataset_name=args.test_dataset,
                             train=False,
                             test=True,
                             batch_size=args.batch_size_val)

    tot_test_x_loader = make_loader(test_dataset,
                                    dataset_name=args.test_dataset,
                                    train=False,
                                    test=False)

    weak_train_z_loader, pos_z_dataset, unl_z_dataset = make_representation(
        encoder,
        weak_train_x_loader,
        with_head=True,
        train=True,
        batch_size = 512)
    
    test_z_dataset = make_representation(encoder, test_loader, with_head=True, train=True, val=True)
    weak_val_z_dataset = make_representation(encoder, weak_val_x_loader, with_head=True, train=True, val=True)
    tot_test_z_dataset = make_representation(encoder, tot_test_x_loader, with_head=True, train=False)
    weak_val_z_loader = DataLoader(weak_val_z_dataset, batch_size=args.batch_size_val, shuffle=False)
    test_z_loader = DataLoader(test_z_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=False)
    


    classifier = Classifier()
    classifier = classifier.cuda()
    
    if args.pretrained:
        print(f'Model loaded from: {args.pretrained}.')
        classifier.load_state_dict(torch.load(args.pretrained))
    else:
        pre_train(classifier, weak_train_z_loader, weak_val_z_loader, test_z_loader, args)

    seed_all(args.seed)

    # Test before train
  
    # val_loss, val_acc = test(classifier, weak_val_z_loader, args)

    # weak_un_loader = DataLoader(unl_z_dataset, batch_size=args.batch_size_val, shuffle=False)
    # un_loss, un_acc = test(classifier, weak_un_loader, args)

    test_loss, test_acc, test_macro_f1 = test(classifier, test_z_loader, args)

    # prYellow(
    #     f'Before training  val-Loss: {val_loss:.5f}  val-Acc: {val_acc * 100.0: .5f}   Test-Acc: {test_acc * 100.0:.5f}')

    history = train(classifier, pos_z_dataset, unl_z_dataset, weak_val_z_dataset, tot_test_z_dataset, test_z_dataset, args)


    test_loss, test_acc, test_macro_1 = test(classifier, test_z_loader, args)
    test_err = 1. - test_acc
    # print(f'Test    Loss: {test_loss}   Error: {100.0 * test_err}')

    return history


def main():
    args = parser.parse_args()
    args.cuda = (not args.no_cuda)
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
    if args.run_all:
        args.debug = False


    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    history, test_loss, test_err = prepare_and_run(args)
    epochs = range(1, len(history['acc']) + 1)


    max_test_acc = float(np.max(history['test_acc']))
    max_test_macro_f1  = float(np.max(history['test_macro_f1']))

    plt.figure(figsize=(10,6))
    plt.plot(epochs, history['acc'],      label='Train Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.plot(epochs, history['test_macro_f1'],  label='Test macro F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(f"seed_{args.seed}/{args.test_dataset}/acc:{max_test_acc:.4f}/f1:{max_test_macro_f1:.4f}")
    plt.legend()
    plt.tight_layout()

    # 기본값: experiment/nnpu → 없으면 자동 생성
    os.makedirs(args.output_dir, exist_ok=True)
    plot_name = f"{args.seed}_{args.src_prior}_{args.tgt_prior}_{args.test_dataset}.png"
    plot_path = os.path.join(args.output_dir, plot_name)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✅  Accuracy curve saved to {plot_path}")


if __name__ == '__main__':
    
    main()