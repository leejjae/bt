import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse, json, math, time, random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from lossFunc import bce_loss, nnpu_loss
from model import build_model
from dataset import DataManager, make_dataset
from utils_fixmatch import make_labels_from_fixmatch

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src_dataset", type=str, default="mnist",
                    choices=["mnist","usps","svhn","cifar","cifarv2","cifar10c","cinic"])
parser.add_argument("--tgt_dataset", type=str, default="mnist",
                    choices=["mnist","usps","svhn","cifar","cifarv2","cifar10c","cinic"])
parser.add_argument("--data_root", type=Path, default=Path("./data"))
parser.add_argument("--src_prior", type=float, default=0.5)
parser.add_argument("--tgt_prior", type=float, default=0.5)

# model / ckpt
parser.add_argument("--arch", type=str, required=True,
                    choices=["lenet","cifar_resnet18","cifar_resnet50","cnn_cifar"])
parser.add_argument("--encoder", type=Path, required=True,
                    help="pretrained encoder (state_dict .pth)")

# training
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--batch_size_val", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu_id", type=int, default=0)

# DST / FixMatch
parser.add_argument("--threshold", type=float, default=0.95, help="FixMatch confidence τ")
parser.add_argument("--T", type=float, default=1.0, help="temperature for pseudo labeling")
parser.add_argument("--trade_off_self_training", type=float, default=1.0)
parser.add_argument("--trade_off_worst", type=float, default=1.0)
parser.add_argument("--mu", type=int, default=1, help="#unlabeled batches per labeled batch")

# logging
parser.add_argument("--out_dir", type=Path, default=Path("./metric/st"))
parser.add_argument("--tag", type=str, default=None)

# -------------------------
# Utils
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, gpu, src_prior):
    model.eval()
    all_logits, all_labels, total_loss, total = [], [], 0.0, 0
    for x, y_true, y in loader:
        x = x.cuda(gpu)
        y_true = y_true.cuda(gpu)
        y = y.cuda(gpu)
        logits = model(x).squeeze(1)              # PU head forward

        y[y==0] = -1
        y_true[y_true==-1] = 0

        loss = nnpu_loss(logits, y, prior=src_prior)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_true.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(np.int64)

    prob = 1 / (1 + np.exp(-all_logits))
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(all_labels, pred)
    f1  = f1_score(all_labels, pred, average="macro")
    try:
        auc = roc_auc_score(all_labels, prob)
    except ValueError:
        auc = float("nan")

    return (total_loss / max(1, total)), acc, f1, auc

# -------------------------
# GRL & Losses (DST)
# -------------------------
class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

class GRL(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam
    def forward(self, x):  # noqa
        return GradReverseFn.apply(x, self.lam)

class WorstCaseEstimationLoss(nn.Module):
    """
    라벨 구간: 주-헤드와 worst-헤드가 일치하도록(soft target BCE)
    비라벨 구간: worst-헤드가 '최악'이 되도록(주-헤드와 불일치 유도; GRL로 특징 쪽은 반대로 학습)
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_l, y_l_adv, y_u, y_u_adv):
        # y_* : logits
        p_l = torch.sigmoid(y_l).detach()          # soft target on labeled
        loss_l = F.binary_cross_entropy_with_logits(y_l_adv, p_l, reduction=self.reduction)

        p_u = torch.sigmoid(y_u).detach()
        # unlabeled에서는 '불일치'를 키우기 위해 음수 부호(adv에는 GRL이 걸려 있으므로 DST 의도와 일치)
        loss_u_align = F.binary_cross_entropy_with_logits(y_u_adv, p_u, reduction=self.reduction)
        loss_worst = loss_l - loss_u_align
        return loss_worst

class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """FixMatch 방식: weak에서 레이블 생성 → strong에 적용"""
    def __init__(self, tau=0.95, T=1.0):
        super().__init__()
        self.tau = tau
        self.T = T
    def forward(self, logits_strong, logits_weak):
        # hard target & mask from weak
        target, mask = make_labels_from_fixmatch(logits_weak, tau=self.tau, T=self.T)
        loss = F.binary_cross_entropy_with_logits(logits_strong, target, reduction="none")
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = logits_strong.new_tensor(0.)
        return loss, mask

# -------------------------
# Model with 3 heads
# -------------------------
class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, freeze: bool, linear_prob: bool):
        super().__init__()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        feat_dim = backbone.feature_dim
        hid = max(32, feat_dim // 4)

        self.fc_pu = nn.Sequential(
            nn.Linear(feat_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
        self.fc_ps = nn.Sequential(
            nn.Linear(feat_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
        self.grl = GRL(1.0)
        self.fc_adv = nn.Sequential(
            nn.Linear(feat_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    # 평가/로그 등에서 편하게 쓰는 PU head 단일 forward
    def forward(self, x):
        z = self.forward_features(x)
        return self.fc_pu(z)

    # DST 학습용: 세 헤드 모두 반환
    def forward_all(self, x):
        z = self.forward_features(x)
        y = self.fc_pu(z)
        y_pseudo = self.fc_ps(z)
        y_adv = self.fc_adv(self.grl(z))
        return y, y_adv, y_pseudo

# -------------------------
# Main
# -------------------------
def main():
    args = parser.parse_args()
    args.rank = 0
    set_seed(args.seed)

    # ----- Data -----
    dm = DataManager(
        train_dataset=args.src_dataset,
        test_dataset=args.tgt_dataset,
        src_prior=args.src_prior,
        tgt_prior=args.tgt_prior
    )
    # 반환 시그니처는 프로젝트 코드에 맞춰 사용
    _, _, train_dataset, val_dataset, test_dataset = dm.get_data(merge=False)

    # 약한 증강(라벨 구간)
    weak_train_x_dataset = make_dataset(train_dataset, args.src_dataset, args.batch_size, role="weak_train")
    weak_val_x_dataset   = make_dataset(val_dataset,   args.src_dataset, args.batch_size_val, role="weak_val")

    # 평가용(normalize만)
    test_dataset_norm    = make_dataset(test_dataset,  args.tgt_dataset, args.batch_size_val, role="test")

    # 타깃(비라벨) weak/strong 쌍
    tot_test_x_dataset   = make_dataset(test_dataset,  args.tgt_dataset, args.batch_size_val, role="tot_test")

    weak_train_x_loader = DataLoader(weak_train_x_dataset, batch_size=args.batch_size,
                                     shuffle=True, drop_last=True)
    weak_val_x_loader   = DataLoader(weak_val_x_dataset, batch_size=args.batch_size_val,
                                     shuffle=False, drop_last=False)
    test_loader         = DataLoader(test_dataset_norm, batch_size=args.batch_size_val,
                                     shuffle=False, drop_last=False)
    tot_test_loader     = DataLoader(tot_test_x_dataset, batch_size=args.batch_size * args.mu,
                                     shuffle=True, drop_last=True)

    # ----- Model & Optim -----
    backbone = build_model(args.arch).cuda(args.gpu_id)
    state_dict = torch.load(args.encoder, map_location="cpu")
    backbone.load_state_dict(state_dict, strict=False)

    model = Classifier(backbone, freeze=True, linear_prob=True).cuda(args.gpu_id)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    st_criterion = ConfidenceBasedSelfTrainingLoss(tau=args.threshold, T=args.T)
    worst_criterion = WorstCaseEstimationLoss()

    best_val_loss = math.inf
    best_state = None
    no_improve = 0
    t0 = time.time()

    # ----- Train -----
    unl_iter = iter(tot_test_loader)
    for epoch in range(args.epochs):
        model.train()
        run_loss, N = 0.0, 0

        for (x_l, y_true_l, y_l) in weak_train_x_loader:
            x_l = x_l.cuda(args.gpu_id)
            y_l = y_l.cuda(args.gpu_id)
            y_true_l = y_true_l.cuda(args.gpu_id)
            y_l[y_l==0] = -1

            # unlabeled batch (weak, strong)
            try:
                (x_u_w, x_u_s), _, _ = next(unl_iter)
            except StopIteration:
                unl_iter = iter(tot_test_loader)
                (x_u_w, x_u_s), _, _ = next(unl_iter)
            x_u_w = x_u_w.cuda(args.gpu_id)
            x_u_s = x_u_s.cuda(args.gpu_id)

            # concat → forward 3 heads (DST 원본 구조)
            x_cat = torch.cat([x_l, x_u_w, x_u_s], dim=0)
            y, y_adv, y_pseudo = model.forward_all(x_cat)

            B = x_l.size(0)

            # 1) PU loss (labeled/source) on PU head
            y_l_logit = y[:B].squeeze(1)
            loss_pu = nnpu_loss(y_l_logit, y_l, prior=args.src_prior)

            # 2) Self-training loss (FixMatch): weak→strong on pseudo head
            y_u_w_logit, _ = y[B:].chunk(2, dim=0)                  # weak on main head
            _, y_u_s_logit = y_pseudo[B:].chunk(2, dim=0)           # strong on pseudo head
            self_training_loss, _ = st_criterion(y_u_s_logit.squeeze(1),
                                                 y_u_w_logit.squeeze(1))
            self_training_loss = args.trade_off_self_training * self_training_loss

            # 3) Worst-case estimation loss (adv head with GRL)
            y_l_adv = y_adv[:B].squeeze(1)
            y_u_adv, _ = y_adv[B:].chunk(2, dim=0)
            y_u_adv = y_u_adv.squeeze(1)
            worst_case_loss = args.trade_off_worst * worst_criterion(
                y_l_logit, y_l_adv, y_u_w_logit.squeeze(1), y_u_adv
            )

            loss = loss_pu + self_training_loss + worst_case_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x_l.size(0)
            N += x_l.size(0)

        train_loss = run_loss / max(1, N)
        scheduler.step()

        # ----- validation on weak_val_x_loader (PU loss)
        model.eval()
        with torch.no_grad():
            val_loss, V = 0.0, 0
            for x, y_true, y in weak_val_x_loader:
                x = x.cuda(args.gpu_id)
                y = y.cuda(args.gpu_id)
                y[y==0] = -1
                logits = model(x).squeeze(1)
                loss = nnpu_loss(logits, y, prior=args.src_prior)
                val_loss += loss.item() * x.size(0)
                V += x.size(0)
            val_loss /= max(1, V)

        # ----- test metrics on test_loader (정규화만)
        te_loss, te_acc, te_f1, te_auc = evaluate(model, test_loader, args.gpu_id, args.src_prior)

        print(f"[epoch {epoch:03d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={te_acc:.4f}  macroF1={te_f1:.4f}  auc={te_auc:.4f}",
              flush=True)

        # early stop
        if val_loss + 1e-9 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(json.dumps({"early_stop_at": epoch}), flush=True)
                break

    # ----- restore best and final test -----
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    te_loss, te_acc, te_f1, te_auc = evaluate(model, test_loader, args.gpu_id, args.src_prior)

    # save
    pair_dir = f"{args.src_dataset}_{args.tgt_dataset}"
    seed_dir = f"seed{args.seed}"
    prior_dir = f"src{args.src_prior}_tgt{args.tgt_prior}"
    arch_dir = f"{args.arch}"
    root_out = args.out_dir / pair_dir / seed_dir / "dst" / prior_dir / arch_dir
    root_out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root_out / "lincls_best.pth")

    result = dict(src=args.src_dataset, tgt=args.tgt_dataset,
                  src_prior=args.src_prior, tgt_prior=args.tgt_prior,
                  arch=args.arch, seed=args.seed,
                  test_loss=te_loss, test_acc=te_acc, test_macro_f1=te_f1, test_auc=te_auc)
    with open(root_out / "results.jsonl", "w") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
