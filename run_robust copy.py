import warnings
warnings.filterwarnings("ignore")

import copy
import json
import argparse
import transformers
import logging
from pathlib import Path
from itertools import cycle

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# import helpers
# import models
import lossFunc
from utils import *
from dataset import DataManager, make_dataset, dataset_to_tensors
from model import build_model
from repr_loader import *
from utils_fixmatch import *



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
parser.add_argument('--lr', default=1e-5, type=float,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--wd', default=0., type=float, help='weight decay (default 0.)')
parser.add_argument('--decay_epoch', default=-1, type=int,
                    help='Reduces the learning rate every decay_epoch (default -1)')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')
parser.add_argument('--cos', action='store_true',
                    help='Use cosine lr scheduler (default False)')
parser.add_argument('--n_warmup', default=5, type=int,
                    help='Number of warm-up steps (default: 0)')
parser.add_argument('--patience', default=999, type=int, help='patience for early stopping (default 5)')
parser.add_argument('--restart', action='store_true',
                    help='reset model before training in each episode (default: False)')

# Test
parser.add_argument('--run_all', action='store_true', help='run all experiences with 20 seeds (default False)')

# CL
parser.add_argument('--inner_epochs', type=int, default=1,
                    help='number of epochs to run after each dataset update (default: 1)')
parser.add_argument('--max_thresh_p', type=float, default=1.5, help='maximum of threshold for labeled (default 2.0)')
parser.add_argument('--max_thresh_n', type=float, default=1.5, help='maximum of threshold for unlabeled (default 2.0)')
parser.add_argument('--grow_steps_p', type=int, default=5, help='number of step to grow to max_thresh for labeled (default 10)')
parser.add_argument('--grow_steps_n', type=int, default=5, help='number of step to grow to max_thresh for unlabeled (default 10)')
parser.add_argument('--scheduler_type_p', type=str, default='linear',
                    help='type of training scheduler for labeled (default linear)')
parser.add_argument('--scheduler_type_n', type=str, default='linear',
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
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--encoder', type=str, required=True)
parser.add_argument("--output_dir", default="experiment/fixrobust", type=str)
args = parser.parse_args()
global_step = 0
moving_weights_all = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def _log_batch_diag(step, tag, logits, labels01):
    """
    step: int (global step or batch idx)
    tag:  str ("train" / "eval")
    logits: (B, 1) or (B,) tensor
    labels01: (B, 1) or (B,) tensor in {0,1}
    """
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    if labels01.dim() > 1:
        labels01 = labels01.squeeze(-1)

    probs = torch.sigmoid(logits)
    pred  = (probs > 0.5).float()

    pos_ratio_in_batch = labels01.mean().item()
    pred_pos_rate      = pred.mean().item()
    logit_mean, logit_std = logits.mean().item(), logits.std().item()
    prob_mean,  prob_std  = probs.mean().item(),  probs.std().item()

    print(f"[{tag} step {step:05d}] "
          f"pos_ratio={pos_ratio_in_batch:.4f}  pred_pos_rate={pred_pos_rate:.4f}  "
          f"logit(m±s)={logit_mean:.4f}±{logit_std:.4f}  prob(m±s)={prob_mean:.4f}±{prob_std:.4f}")


# def evaluate(model, test_dataset, args):
#     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

#     model.eval()
#     # loss = lossFunc.bce_loss

#     ys, ps = [], []
#     loss_sum, n_sum = 0.0, 0
#     with torch.no_grad():
#         for x, y_true, _ in test_loader:
#             x = x.cuda(args.gpu_id)
#             y_true = y_true.cuda(args.gpu_id).float()
#             y_true[y_true==-1]=0
#             logits = model(x).squeeze(1)

#             loss = lossFunc.bce_loss(logits, y_true)
#             loss_sum += loss.item() * x.size(0)
#             n_sum += x.size(0)

#             probs = torch.sigmoid(logits).detach().cpu().numpy()
#             ps.append(probs)
#             ys.append(y_true.detach().cpu().numpy())

#     y_np = np.concatenate(ys, 0).astype(int)
#     p_np = np.concatenate(ps, 0)

#     loss = (loss_sum / max(1, n_sum))
#     pred = (p_np >= 0.5).astype(int)
#     acc = accuracy_score(y_np, pred)
#     f1  = f1_score(y_np, pred, average="macro")

#     try: 
#         auc = roc_auc_score(y_np, p_np)
#     except ValueError: 
#         auc = float("nan")
#     return loss, acc, f1, auc


def evaluate(model, test_dataset, args):
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

    model.eval()
    ys, ps, ls = [], [], []   # ← logits 수집 리스트 추가
    loss_sum, n_sum = 0.0, 0
    with torch.no_grad():
        for x, y_true, _ in test_loader:
            x = x.cuda(args.gpu_id)
            y_true = y_true.cuda(args.gpu_id).float()
            y_true[y_true==-1]=0
            logits = model(x).squeeze(1)

            loss = lossFunc.bce_loss(logits, y_true)
            loss_sum += loss.item() * x.size(0)
            n_sum += x.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            ps.append(probs)
            ys.append(y_true.detach().cpu().numpy())
            ls.append(logits.detach().cpu().numpy())        # ← 추가

    y_np = np.concatenate(ys, 0).astype(int)
    p_np = np.concatenate(ps, 0)
    l_np = np.concatenate(ls, 0).ravel()                    # ← 추가

    loss = (loss_sum / max(1, n_sum))
    pred = (p_np >= 0.5).astype(int)
    acc = accuracy_score(y_np, pred)
    f1  = f1_score(y_np, pred, average="macro")

    try:
        auc = roc_auc_score(y_np, p_np)
    except ValueError:
        auc = float("nan")

    # === 부호 뒤집힘 진단 ===
    try:
        auc_pos = roc_auc_score(y_np, l_np)
        auc_neg = roc_auc_score(y_np, -l_np)
        print(f"[eval AUC check] AUC(logits)={auc_pos:.4f}  AUC(-logits)={auc_neg:.4f}")
    except Exception as e:
        print("[eval AUC check] failed:", e)

    return loss, acc, f1, auc



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
                data, true_labels, labels = data.cuda(), true_labels.cuda(), labels.cuda()

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

        # unlabel_weights = moving_weights_all[labels_all == -1]
        # unlabel_true_labels = true_labels_all[labels_all == -1]
        # unlabel_probs = probs_all[labels_all == -1]
        

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

            
    classifier.train()
    test_iter = cycle(test_loader)

    tot_loss_sum = 0.0
    tot_true_loss_sum = 0.0
    tot_acc_sum = 0.0
    tot_cnt = 0
    
    for inner_epoch in range(args.inner_epochs):
        for step, (data, true_labels, labels, weights) in enumerate(weighted_train_dataloader):
            (weak_test_z, strong_test_z), _, _ = next(test_iter)
            if args.cuda:
                data, true_labels, labels,  weights = data.cuda(), true_labels.cuda(), labels.cuda(), weights.cuda()
                weak_test_z, strong_test_z = weak_test_z.cuda(non_blocking=True), strong_test_z.cuda(non_blocking=True)
            
            labels = (labels > 0).float().unsqueeze(1)
            true_labels = (true_labels > 0).float().unsqueeze(1)

            batch_ = torch.cat([data, weak_test_z, strong_test_z])
            batch = interleave(batch_, 2*args.mu + 1)

            logits = classifier(batch)
            logits = de_interleave(logits, 2*args.mu + 1)

            B = data.size(0)
            logits_x = logits[:B]
            logits_u_w, logits_u_s = logits[B:].chunk(2)
            del logits

            if step % 50 == 0:
                _log_batch_diag(step, "train", logits_x.detach(), labels.detach())

            # loss w.r.t. pseudo labels
            if args.loss == 'bce':
                loss_src = lossFunc.bce_loss(logits_x, labels, weights)
            elif args.loss == 'focal':
                loss_src = lossFunc.b_focal_loss(logits_x, labels, weights, gamma=args.focal_gamma)
            else:
                loss_src = getattr(lossFunc, f'{args.loss}_loss')(logits_x, labels, args.src_prior, weights)


            target, mask = make_labels_from_fixmatch(logits_u_w, tau=0.98, T=1)
            loss_tgt = F.binary_cross_entropy_with_logits(logits_u_s, target, reduction='none')
            loss_tgt = (loss_tgt * mask).mean()

            loss = loss_src + args.lam_u * loss_tgt
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss w.r.t. true labels
            true_loss = lossFunc.bce_loss(logits_x, true_labels, weights)
            # acc w.r.t. true labels
            tot_loss_sum += loss.item() * data.size(0)
            tot_true_loss_sum += true_loss.item() * data.size(0)
            tot_cnt += data.size(0)

        scheduler.step()
        
   
    return (tot_loss_sum/max(1,tot_cnt)), (tot_acc_sum/max(1,tot_cnt)), (tot_true_loss_sum/max(1,tot_cnt))


def train(classifier, positive_dataset, unlabeled_dataset, val_dataset, tot_test_dataset, test_dataset, args):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False ,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False ,drop_last=True)
    tot_test_loader = DataLoader(tot_test_dataset, batch_size=args.batch_size_val*args.mu, shuffle=False ,drop_last=True)

    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience

    pos_tensor = dataset_to_tensors(positive_dataset, batch_size=batch_size)
    unl_tensor = dataset_to_tensors(unlabeled_dataset, batch_size=batch_size)

    # 전체 텐서 꺼내기: (X, Y_true, Y)
    pos_X, pos_Yt, pos_Y = pos_tensor.tensors
    unl_X, unl_Yt, unl_Y = unl_tensor.tensors

    # unlabeled 라벨 정리: 0 -> -1
    # (원본 유지 원하면 clone() 후 수정)
    unl_Y = unl_Y.clone()
    unl_Y[unl_Y == 0] = -1

    # 최종 학습 입력
    positive_data   = pos_X
    positive_labels = pos_Y            # (보통 1)
    unlabeled_data  = unl_X
    unlabeled_labels= unl_Y   


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

    best_val_loss = float('inf')
    no_improve = 0


    val_best_model = copy.deepcopy(classifier.state_dict())

    fea_all = []
    history_test_acc = []
    history_test_macro_f1 = []
    history_test_auc = [] 
    for episode in range(epochs): 
        
        # get next lambda
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

       
        tot_loss, tot_acc, tot_true_loss = train_episode(classifier, weighted_train_loader, tot_test_loader, args)


        val_loss, val_acc, val_macro_f1, val_auc = evaluate(classifier, val_dataset, args)
        test_loss, test_acc, test_macro_f1, test_auc = evaluate(classifier, test_dataset, args)
        
        print(f"[epoch {episode:03d}] "
        f"train={tot_loss:.4f}  val={val_loss:.4f}  "
        f"acc={test_acc:.4f}  macroF1={test_macro_f1:.4f}  auc={test_auc:.4f}",
        flush=True)

        history_loss.append(tot_loss)
        history_acc.append(tot_acc)
        history_true_loss.append(tot_true_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        history_test_acc.append(test_acc)
        history_test_macro_f1.append(test_macro_f1)
        history_test_auc.append(test_auc)


        if val_loss + 1e-9 < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            val_best_model = copy.deepcopy(classifier.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                break
            classifier.load_state_dict(val_best_model)

    
    history = {'pseudo_loss': history_loss, 'true_loss': history_true_loss, 'acc': history_acc,
               'val_loss': history_val_loss, 'val_acc': history_val_acc,
               'test_acc' : history_test_acc, 'test_macro_f1': history_test_macro_f1, 'test_auc':history_test_auc}

    return history



class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, freeze: bool, linear_prob: bool):
        super().__init__()
        self.backbone = backbone
        self.linear_prob = linear_prob
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        input_dim = backbone.feature_dim
        hidden_dim = int(input_dim/4)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        z = self.backbone.forward_features(x)
        return self.fc(z)



def prepare_and_run(args):
    seed_all(args.seed)

    # backbone = build_model(args.arch)
    # backbone = backbone.cuda(args.gpu_id)
    # state_dict = torch.load(args.encoder, map_location='cpu')
    # backbone.load_state_dict(state_dict, strict=False)


    data_manger = DataManager(train_dataset=args.train_dataset,
                              test_dataset=args.test_dataset,
                              src_prior=args.src_prior,
                              tgt_prior=args.tgt_prior)
    

    pos_dataset, unl_dataset, val_dataset, test_dataset = data_manger.get_data(merge=False)

    
    pos_x_dataset = make_dataset(pos_dataset, args.train_dataset, args.batch_size, role='weak_train')
    unl_x_dataset = make_dataset(unl_dataset, args.train_dataset, args.batch_size, role='weak_train')


    # weak_train_x_dataset = make_dataset(train_dataset, args.train_dataset, args.batch_size, role='weak_train')
    weak_val_x_dataset = make_dataset(val_dataset, args.train_dataset, args.batch_size_val, role='weak_val')
    test_dataset = make_dataset(test_dataset, args.test_dataset, args.batch_size_val, role='test')
    tot_test_x_dataset = make_dataset(test_dataset, args.train_dataset, args.batch_size_val, role='tot_test')

    backbone = build_model(args.arch)
    backbone = backbone.cuda(args.gpu_id)
    state_dict = torch.load(args.encoder, map_location='cpu')
    backbone.load_state_dict(state_dict, strict=False)
    classifier = Classifier(backbone, freeze=True, linear_prob=True).cuda(args.gpu_id)

    history = train(classifier,
                    pos_x_dataset,
                      unl_x_dataset,
                      weak_val_x_dataset,
                      tot_test_x_dataset,
                      test_dataset,
                      args)
    test_loss, test_acc, test_macro_f1, test_auc = evaluate(classifier, test_dataset, args)
    pair_dir = f"{args.train_dataset}_{args.test_dataset}"
    seed_dir = f"seed{args.seed}"
    loss_dir = f"loss{args.loss}"
    prior_dir = f"src{args.src_prior}_tgt{args.tgt_prior}"
    arch_dir = f"{args.arch}"
    root_out = Path(args.output_dir) / pair_dir / seed_dir / loss_dir / prior_dir / arch_dir
    root_out.mkdir(parents=True, exist_ok=True)

    # ===== results.jsonl 저장 =====
    result_line = dict(
        src=args.train_dataset, tgt=args.test_dataset,
        src_prior=args.src_prior, tgt_prior=args.tgt_prior,
        arch=args.arch, seed=args.seed, loss=args.loss,
        test_loss=float(test_loss), test_acc=float(test_acc),
        test_macro_f1=float(test_macro_f1), test_auc=float(test_auc)
    )
    with open(root_out / "results.jsonl", "w") as f:
        f.write(json.dumps(result_line) + "\n")


    return history


def main():
    args = parser.parse_args()
    args.cuda = (not args.no_cuda)
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
    if args.run_all:
        args.debug = False

    pair_dir = f"{args.train_dataset}_{args.test_dataset}"
    seed_dir = f"seed{args.seed}"
    loss_dir = f"loss{args.loss}"
    prior_dir = f"src{args.src_prior}_tgt{args.tgt_prior}"
    arch_dir = f"{args.arch}"
    root_out = Path(args.output_dir) / pair_dir / seed_dir / loss_dir / prior_dir / arch_dir
    root_out.mkdir(parents=True, exist_ok=True)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    history = prepare_and_run(args)
    epochs = range(1, len(history['acc']) + 1)


    plt.figure(figsize=(10,6))
    # plt.plot(epochs, history['acc'],      label='Train Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.plot(epochs, history['test_macro_f1'],  label='Test macro F1-score')
    plt.plot(epochs, history['test_auc'],  label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title(f"seed_{args.seed}/{args.test_dataset}")
    plt.legend()
    plt.tight_layout()

    # 기본값: experiment/nnpu → 없으면 자동 생성
    plt.figure(figsize=(10,6))
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.plot(epochs, history['test_macro_f1'], label='Test macro F1-score')
    plt.plot(epochs, history['test_auc'], label='Test AUC')
    plt.xlabel('Epoch'); plt.ylabel('Metric')
    plt.title(f"seed_{args.seed}/{args.test_dataset}")
    plt.legend(); plt.tight_layout()
    plt.savefig(root_out / "metric.png", dpi=200)
    plt.close()
    print(f"✅ curves saved to {root_out / 'curves.png'}")


if __name__ == '__main__':
    
    main()