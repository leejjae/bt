import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import sys
import time
import urllib

from copy import deepcopy
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import numpy as np
import torchvision

from lossFunc import bce_loss, nnpu_loss
from model import build_model
from dataset import DataManager

parser = argparse.ArgumentParser()
parser.add_argument("--src_dataset", type=str, default='mnist',
                        choices=["mnist","usps","svhn","cifar","cifarv2","cifar10c","cinic"])
parser.add_argument("--tgt_dataset", type=str, default='mnist',
                        choices=["mnist","usps","svhn","cifar","cifarv2","cifar10c","cinic"])
parser.add_argument("--data_root", type=Path, default=Path("./data"))
parser.add_argument("--src_prior", type=float, default=0.5)
parser.add_argument("--tgt_prior", type=float, default=0.5)
parser.add_argument("--num_labeled", type=int, default=1000)
parser.add_argument("--num_unlabeled", type=int, default=5000)

# model/ckpt
parser.add_argument("--arch", type=str, required=True,
                        choices=["lenet","cifar_resnet18","cifar_resnet50","cnn_cifar"])
parser.add_argument("--encoder", type=Path, required=True, help="pretrained encoder (state_dict .pth)")

# training
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--loss_type", type=str, default='pu')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
parser.add_argument("--val_split", type=float, default=0.1, help="fraction of src for validation")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu_id", type=int, default=0)

# logging/saving
parser.add_argument("--out_dir", type=Path, default=Path("./metric/lin"))
parser.add_argument("--tag", type=str, default=None) 

    

def main():
    args = parser.parse_args()
    args.rank = 0
    set_seed(args.seed)

    data_manager = DataManager(
        train_dataset = args.src_dataset,
        test_dataset = args.tgt_dataset,
        src_prior = args.src_prior,
        tgt_prior = args.tgt_prior
        )
    src_dataset, tgt_dataset = data_manager.get_data(merge=True)
    src_transform = get_transforms(args.src_dataset, args.arch)
    tgt_transform = get_transforms(args.tgt_dataset, args.arch)

    src_dataset.transform = src_transform

    tgt_dataset.transform = tgt_transform


    val_len = int(len(src_dataset) * args.val_split)
    train_len = len(src_dataset) - val_len
    tr_set, val_set = random_split(
        src_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    backbone = build_model(args.arch)
    backbone = backbone.cuda(args.gpu_id)
    state_dict = torch.load(args.encoder, map_location='cpu')
    backbone.load_state_dict(state_dict, strict=False)

    
    model = Classifier(backbone, freeze=True, linear_prob=True).cuda(args.gpu_id)
    ema_model = ModelEMA(model, decay=0.999) if args.loss_type == 'bce' else None
    if args.loss_type =='bce':
        ema_model = ModelEMA(model, decay=0.999)
    
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_loss = math.inf
    best_state = None
    patience = args.patience
    no_improve = 0
    start_time = time.time()


    for epoch in range(args.epochs):
        model.train()
        total = 0
        run_loss = 0.0
        for x, y_true, y in train_loader:
            x = x.cuda(args.gpu_id)
            y = y.cuda(args.gpu_id)
            y_true = y_true.cuda(args.gpu_id)
            y[y==0]=-1
            logits = model(x).squeeze(1)

            if args.loss_type == 'pu':
                loss = nnpu_loss(logits, y, prior=args.src_prior)
            else:
                weights = torch.where(
                    y == 1,
                    torch.ones_like(y, dtype=torch.float32, device=y.device),
                    torch.full_like(y, 0.1, dtype=torch.float32, device=y.device)
                )
                loss = bce_loss(logits, y, weights=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)

            run_loss += loss.item() * x.size(0)
            total += x.size(0)

        tr_loss = run_loss / max(1, total)

        # validation
        # evaluate(model, loader, gpu, loss_type, src_prior)
        val_model = ema_model if ema_model is not None else model
        val_loss, val_acc, val_f1, val_auc = evaluate(
            val_model,
            val_loader,
            args.gpu_id,
            args.loss_type,
            args.src_prior)
        
        test_loss, test_acc, test_f1, test_auc = evaluate(
            val_model,
            test_loader,
            args.gpu_id,
            args.loss_type,
            args.src_prior)

        # 로그
        log = dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss,
                   val_acc=val_acc, val_macro_f1=val_f1, val_auc=val_auc,
                   time=int(time.time()-start_time))
        
        print(f"[epoch {epoch:03d}] "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"acc={test_acc:.4f}  macroF1={test_f1:.4f}  auc={test_auc:.4f}",
              flush=True)

        # early stopping 체크
        improved = val_loss + 1e-9 < best_val_loss
        if improved:
            best_val_loss = val_loss
            snapshot_src = val_model  # EMA가 있으면 EMA, 없으면 원모델
            best_state = {k: v.detach().cpu() for k, v in snapshot_src.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(json.dumps({"early_stop_at": epoch}), flush=True)
                break

        scheduler.step()

    # ---------------------------------------------------------------------
    # 4) 최종: best로 복원 → tgt(test) 평가
    # ---------------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    # evaluate(model, loader, gpu, loss_type, src_prior)
    test_model = ema_model if ema_model is not None else model
    te_loss, te_acc, te_f1, te_auc = evaluate(test_model, test_loader, args.gpu_id, args.loss_type, args.src_prior)



    pair_dir = f"{args.src_dataset}_{args.tgt_dataset}"
    seed_dir = f"seed{args.seed}"
    loss_dir = f"loss{args.loss_type}"
    prior_dir = f"src{args.src_prior}_tgt{args.tgt_prior}"
    arch_dir = f"{args.arch}"

    root_out = args.out_dir / pair_dir / seed_dir / loss_dir / prior_dir / arch_dir
    root_out.mkdir(parents=True, exist_ok=True)

    # (원하시던 단순 tag는 그냥 유지; 로그 출력 등에만 쓰고 파일명엔 사용 X)
    tag = args.tag or f"src{args.src_prior}_tgt{args.tgt_prior}_{args.arch}"

    # === 가중치/결과 저장 (파일명 고정) ===
    torch.save(
        best_state if best_state is not None else model.state_dict(),
        root_out / "lincls_best.pth"
    )

    result_line = dict(
        src=args.src_dataset, tgt=args.tgt_dataset,
        src_prior=args.src_prior, tgt_prior=args.tgt_prior,
        arch=args.arch, seed=args.seed, loss=args.loss_type,
        test_loss=te_loss, test_acc=te_acc, test_macro_f1=te_f1, test_auc=te_auc
    )

    with open(root_out / "results.jsonl", "w") as f:
        f.write(json.dumps(result_line) + "\n")

    print(json.dumps(result_line), flush=True)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_transforms(dataset: str, arch: str):
    ds = dataset.lower()
    arch = arch.lower()

    if arch == "lenet":
        # LeNet: 1ch, 28x28
        size = 28
        out_channels = 1
        mean, std = [0.1307], [0.3081]
    else:
        # CIFAR 계열: 3ch, 32x32
        size = 32
        out_channels = 3
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2470, 0.2435, 0.2616]

    transforms_lst = []
    # 채널 강제 (그레이스케일/컬러 모두 아키텍처에 맞춤)
    if out_channels == 1:
        transforms_lst.append(transforms.Grayscale(num_output_channels=1))
    else:
        transforms_lst.append(transforms.Grayscale(num_output_channels=3))
    # 해상도 강제
    transforms_lst.append(transforms.Resize(size, interpolation=InterpolationMode.BILINEAR))
    transforms_lst.append(transforms.ToTensor())
    transforms_lst.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transforms_lst)




@torch.no_grad()
def evaluate(model, loader, gpu, loss_type, src_prior):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0
    total = 0
    for x, y_true, y in loader:
        x = x.cuda(gpu)
        y = y.cuda(gpu)
        y_true = y_true.cuda(gpu)
        logits = model(x).squeeze(1)

        y_true[y_true==-1]=0
        y[y==0]=-1

        if loss_type == 'pu':
                loss = nnpu_loss(logits, y, prior=src_prior)
        else:
            weights = torch.where(
                y == 1,
                torch.ones_like(y, dtype=torch.float32, device=y.device),
                torch.full_like(y, 0.1, dtype=torch.float32, device=y.device)
            )
            loss = bce_loss(logits, y, weights=weights)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_true.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(np.int64)

    # metrics
    prob = 1 / (1 + np.exp(-all_logits))  # sigmoid
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(all_labels, pred)
    f1  = f1_score(all_labels, pred, average="macro")

    # AUC은 양/음 한 클래스만 있을 때 에러 → 방어
    try:
        auc = roc_auc_score(all_labels, prob)
    except ValueError:
        auc = float("nan")

    avg_loss = total_loss / max(1, total)
    return avg_loss, acc, f1, auc




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

class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = float(decay)
        self.ema_model = deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, base_model: nn.Module):
        for p_src, p_ema in zip(base_model.parameters(), self.ema_model.parameters()):
            if not p_src.requires_grad:
                continue
            p_ema.data.mul_(self.decay).add_(p_src.data, alpha=1.0 - self.decay)

    def forward(self, x):
        return self.ema_model(x)

    def to(self, *args, **kwargs):
        self.ema_model.to(*args, **kwargs)
        return self

    def eval(self):
        self.ema_model.eval()
        return super().eval()




if __name__ == '__main__':
    main()