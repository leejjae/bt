import warnings
warnings.filterwarnings("ignore")

import os
import sys
import math
import torch
import random
import numpy as np
import random
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset

###########################
### utils about CL data ###
###########################

cifar_avg = [0.4914, 0.4822, 0.4465]
cifar_std  = [0.2470, 0.2435, 0.2616]
mnist_avg = [0.1307]
mnist_std  = [0.3081]

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PairTransform:
    def __init__(self, size: int, three_channels: bool):
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        normalize = transforms.Normalize(
            mean=cifar_avg if three_channels else mnist_avg,
            std=cifar_std if three_channels else mnist_std
        )

        to_3ch = [] if three_channels else [mnist_std.Grayscale(num_output_channels=1)]
        # three_channels=True인데 원본이 1ch일 수 있으므로 3ch로 복제
        ensure_3ch = [transforms.Grayscale(num_output_channels=3)] if three_channels else []

        base = [
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        self.t1 = transforms.Compose(
            ensure_3ch + base + [
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),                  # 강 blur
                Solarization(p=0.0),
                transforms.ToTensor(),
                normalize
            ]
        )
        self.t2 = transforms.Compose(
            ensure_3ch + base + [
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),                    # 약 blur
                Solarization(p=0.2),
                transforms.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.t1(x), self.t2(x)



class TwoViewDataset(Dataset):
    def __init__(self, base, pair_transform: PairTransform):
        self.base = base
        self.pair_t = pair_transform
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        img = item[0]
        y1, y2 = self.pair_t(img)
        label = item[1] if len(item) > 1 else -1
        return (y1, y2), label, idx
    


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())



def euclidean_distance(X, v):
    return torch.sum(torch.square(X - v), dim=-1)



class TrainingScheduler:
    def __init__(self, TS_type, init_ratio, max_thresh, grow_steps, p=None, eta=None, lam=0.5):
        super(TrainingScheduler, self).__init__()
        self.init_ratio = init_ratio
        self.type = TS_type
        self.p = p
        self.eta = eta
        self.step = 0
        self.grow_steps = grow_steps
        self.max_thresh = max_thresh
        self.lam = lam
        # assert 0.0 <= self.init_ratio <= self.max_thresh
        self.cal_lib = torch if isinstance(self.init_ratio, torch.Tensor) else math

    def get_next_ratio(self):
        if self.type == 'const':
            ratio = self.init_ratio
        elif self.type == 'linear':
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) / self.grow_steps * self.step
        elif self.type == 'convex':  # from fast to slow
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * self.cal_lib.sin(self.step / self.grow_steps * np.pi * 0.5)
        elif self.type == 'concave':  # from slow to fast
            if self.step > self.grow_steps:
                ratio = self.max_thresh
            else:
                ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * (1. - self.cal_lib.cos(self.step / self.grow_steps * np.pi * 0.5))
        elif self.type == 'exp':
            assert 0 <= self.lam <= 1
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * (1. - self.lam ** self.step)
        else:
            raise NotImplementedError(f'Invalid Training Scheduler type {self.type}')

        if self.init_ratio < self.max_thresh:
            ratio = min(ratio, self.max_thresh)
        else:
            ratio = max(ratio, self.max_thresh)
        self.step += 1
        return ratio


def calculate_spl_weights(x, thresh, args, eps=1e-1):
    spl_type = args.spl_type
    assert thresh > 0., 'spl threshold must be positive'
    if spl_type == 'hard':
        # assert 0. <= thresh <= 1.
        # thresh = torch.quantile(x, thresh)
        weights = (x < thresh).float()
    elif spl_type == 'linear':
        weights = 1. - x / thresh
        weights[x >= thresh] = 0.
    elif spl_type == 'log':
        thresh = min(thresh, 1. - eps)
        assert 0. < thresh < 1., 'Logarithmic need thresh in (0, 1)'
        weights = torch.log(x + 1. - thresh) / torch.log(torch.tensor(1. - thresh))
        weights[x >= thresh] = 0.
    elif spl_type == 'mix2':
        gamma = args.mix2_gamma
        weights = gamma * (1. / torch.sqrt(x) - 1. / thresh)
        weights[x <= (thresh * gamma / (thresh + gamma)) ** 2] = 1.
        weights[x >= thresh ** 2] = 0.
    elif spl_type == 'logistic':
        weights = (1. + torch.exp(torch.tensor(-thresh))) / (1. + torch.exp(x - thresh))
    elif spl_type == 'poly':
        t = args.poly_t
        assert t > 1, 't in polynomial must > 1'
        weights = torch.pow(1. - x / thresh, 1. / (t - 1))
        weights[x >= thresh] = 0.
    elif spl_type == 'welsch':
        weights = torch.exp(-x / (thresh * thresh))
    elif spl_type == 'cauchy':
        weights = 1. / (1. + x / (thresh * thresh))
    elif spl_type == 'huber':
        sx = torch.sqrt(x)
        weights = thresh / sx
        weights[sx <= thresh] = 1.
    elif spl_type == 'l1l2':
        # thresh must decrease when using L1L2
        weights = 1. / torch.sqrt(thresh + x)
    else:
        raise ValueError('Invalid spl_type')
    assert weights.min() >= 0. - eps and weights.max() <= 1. + eps, f'weight [{weights.min()}, {weights.max()}] must in range [0., 1.]'
    return weights