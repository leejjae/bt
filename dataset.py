import os
import copy
import random
import scipy.io
from typing import List
from collections import namedtuple

import numpy as np
import torch


from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image, InterpolationMode
from sklearn.utils import shuffle
from randaugment import RandAugmentMC

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = os.path.join(os.path.dirname(__file__), 'data/')
binary_class_mapping = {
    'comp_rec':{'pos_classes': [0,1,2,3,4], 'neg_classes':[5,6,7,8]},
    'comp_sci':{'pos_classes': [0,1,2,3,4], 'neg_classes':[5,6,7,8]},
    'comp_talk':{'pos_classes': [0,1,2,3,4], 'neg_classes':[5,6,7,8]},
    'mnist': {'pos_classes': [0,2,4,6,8], 'neg_classes':[1,3,5,7,9]},
    'cifar': {'pos_classes': [0,1,8,9], 'neg_classes': [2,3,4,5,6,7]},
    'svhn':  {'pos_classes': [0,2,4,6,8], 'neg_classes': [1,3,5,7,9]},
    'usps':  {'pos_classes': [0,2,4,6,8], 'neg_classes': [1,3,5,7,9]},
    'cifarv2':{'pos_classes':[0,1,8,9], 'neg_classes': [2,3,4,5,6,7]},
    'cifar10c':{'pos_classes':[0,1,8,9], 'neg_classes': [2,3,4,5,6,7]},
    'cinic':{'pos_classes':[0,1,8,9], 'neg_classes': [2,3,4,5,6,7]},
}

class DataManager:
    def __init__(self,
                 train_dataset,
                 test_dataset,
                 num_labeled=1000,
                 num_unlabeled=5000,
                 src_prior=0.5,
                 tgt_prior=0.5,
                 batch_size=512,
                 device=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = batch_size
        self.test_batch_size = batch_size
        self.device = device

        ## attributes specific to dataset
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.src_prior = src_prior
        self.tgt_prior = tgt_prior
        self.neg_classes, self.pos_classes = None, None  # PU specific
        self.num_classes = None
        self.num_channels, self.height, self.width = None, None, None
        self.tr_dataset, self.te_dataset = None, None
        self.sv_transform, self.mv_transform, self.basic_transform = None, None, None

        
        self.dataset_map = {
            'mnist': BinaryMNIST,     
            "svhn" : BinarySVHN,      
            'usps' : BinaryUSPS,      
            'cifar': BinaryCIFAR10,  
            'cifarv2': BinaryCIFAR10v2,
            'cifar10c' : BinaryCIFAR10C,
            'cinic' : BinaryCINICIMGNET
        }
   


    def get_data(self, merge):
        
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset

        self.tr_pos_class = binary_class_mapping[train_dataset]['pos_classes']
        self.tr_neg_class = binary_class_mapping[train_dataset]['neg_classes']
        self.te_pos_class = binary_class_mapping[test_dataset]['pos_classes']
        self.te_neg_class = binary_class_mapping[test_dataset]['neg_classes']

        tr_dataset = self.dataset_map[train_dataset](
            pos_class = self.tr_pos_class,
            neg_class = self.tr_neg_class,
            num_labeled = self.num_labeled,
            num_unlabeled = self.num_unlabeled,
            src_prior = self.src_prior
        )
       
        # Test Data => covariate shift
        if test_dataset =='cifarv2':
            npz_path = './data/cifar10v2/cifar102_test.npz'
            te_dataset = self.dataset_map[test_dataset](
                 npz_file = npz_path,
                 pos_class = self.te_pos_class,
                 neg_class = self.te_neg_class,
                 num_labeled = 0,
                 num_unlabeled = self.num_unlabeled,
                 tgt_prior = self.tgt_prior
            )
            
        elif test_dataset == 'cifar10c':
             npz_path = './data/CIFAR-10-C'
             te_dataset = self.dataset_map[test_dataset](
                 pos_class = self.te_pos_class,
                 neg_class = self.te_neg_class,
                 num_labeled = 0,
                 num_unlabeled = self.num_unlabeled,
                 tgt_prior = self.tgt_prior
            )
            
        elif test_dataset in ['svhn', 'usps', 'cinic']:
             te_dataset = self.dataset_map[test_dataset](
                 pos_class = self.te_pos_class,
                 neg_class = self.te_neg_class,
                 num_labeled = 0,
                 num_unlabeled = self.num_unlabeled,
                 tgt_prior = self.tgt_prior
            )
            
        if merge:
            return tr_dataset, te_dataset

        else:
            y = tr_dataset.y

            pos_idx = np.where(y == 1)[0]        # labeled positives
            unl_idx = np.where(y != 1)[0]        # unlabeled (y==0 or -1)

            # Validation set (ex: 전체의 10%)
            total_len = len(tr_dataset)
            val_len = int(total_len * 0.1)
            train_len = total_len - val_len
            tr_subset, val_subset = random_split(
                tr_dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42)
            )

            # pos/unl dataset은 Dataset 객체로 따로 만들어서 반환
            pos_dataset = torch.utils.data.Subset(tr_dataset, pos_idx.tolist())
            unl_dataset = torch.utils.data.Subset(tr_dataset, unl_idx.tolist())
            val_dataset = val_subset
            test_dataset = te_dataset

            return pos_dataset, unl_dataset, tr_dataset, val_dataset, test_dataset
    


class BinaryCIFAR10(datasets.CIFAR10):

    def __init__(self,
                 pos_class: List,
                 neg_class: List = None,
                 root=root_dir,
                 train: bool = True,
                 num_labeled: int = None,
                 num_unlabeled: int = None,
                 src_prior: float = 0.5):
        super().__init__(root=root, train=train, download=True)
        self.data, self.targets = np.array(self.data), np.array(self.targets)
        self.multiclass_target = self.targets
        self.data, self.y_true, self.y = binarize_dataset(
            features=self.data,
            targets=self.targets,
            pos_class=pos_class,
            neg_class=neg_class,
            num_labeled=num_labeled,
            num_unlabeled=num_unlabeled,
            prior=src_prior
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        img, y_true, y = self.data[idx], self.y_true[idx], self.y[idx]

        img = Image.fromarray(img.astype(np.uint8))
    
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, y_true, y

    

class BinaryMNIST(datasets.MNIST):
    def __init__(
        self,
        pos_class: List,
        neg_class: List = None,
        root=root_dir,
        train: bool = True,
        num_labeled: int = None,
        num_unlabeled: int = None,
        src_prior: float = 0.5,
      
    ):
        super().__init__(root=root, train=train, download=True)
        self.data, self.targets = np.array(self.data), np.array(self.targets)
        self.multiclass_targets = self.targets
        self.data, self.y_true, self.y = binarize_dataset(
            features=self.data,
            targets=self.targets,
            pos_class=pos_class,
            neg_class=neg_class,
            num_labeled=num_labeled,
            num_unlabeled=num_unlabeled,
            prior=src_prior
        )
        # self.data = torch.from_numpy(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        img, y_true, y = self.data[idx], self.y_true[idx], self.y[idx]

        img = Image.fromarray(img.astype(np.uint8))
    
        if self.transform is not None:
            img = self.transform(img)
        # else:
        #     img = transforms.ToTensor()(img)

        return img, y_true, y  


class BinarySVHN(Dataset):
    def __init__(
        self,
        pos_class: List,
        neg_class: List = None,
        root=root_dir,
        train: bool = True,
        num_labeled: int = None,
        num_unlabeled: int = None,
        tgt_prior: float = 0.5,
        transform=None
    ):
        
        svhn_data = scipy.io.loadmat("data/test_32x32.mat")
        self.data = svhn_data['X']
        self.data = np.transpose(self.data, (3,0,1,2))
        self.targets = svhn_data['y']
        self.targets[self.targets==10] = 0

        self.multiclass_targets = self.targets
        self.data, self.y_true, self.y = binarize_dataset(
            features=self.data,
            targets=self.targets,
            pos_class=pos_class,
            neg_class=neg_class,
            num_labeled=num_labeled,
            num_unlabeled=num_unlabeled,
            prior=tgt_prior
        )
        self.transform = transform
    def __len__(self):
         return len(self.data)
    
    def __getitem__(self, idx):
        img, y_true, y = self.data[idx], self.y_true[idx], self.y[idx]
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, y_true, y 


class BinaryUSPS(datasets.USPS):
    def __init__(
        self,
        pos_class: List,
        neg_class: List = None,
        root=root_dir,
        train: bool = True,
        num_labeled: int = None,
        num_unlabeled: int = None,
        tgt_prior: float = 0.5
    ):
        super().__init__(root=root, train=train, download=True)
        self.data, self.targets = np.array(self.data), np.array(self.targets)
        self.multiclass_targets = self.targets
        self.data, self.y_true, self.y = binarize_dataset(
            features=self.data,
            targets=self.targets,
            pos_class=pos_class,
            neg_class=neg_class,
            num_labeled=num_labeled,
            num_unlabeled=num_unlabeled,
            prior=tgt_prior
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        img, y_true, y = self.data[idx], self.y_true[idx], self.y[idx]

        img = Image.fromarray(img.astype(np.uint8))
    
        if self.transform is not None:
            img = self.transform(img)
        # else:
        #     img = transforms.ToTensor()(img)

        return img, y_true, y   



class BinaryCIFAR10v2(Dataset):
    def __init__(
        self,
        npz_file: str,
        pos_class: List,
        neg_class: List = None,
        root=root_dir,
        train: bool = True,
        num_labeled: int = None,
        num_unlabeled: int = None,
        tgt_prior: float = 0.5,
        transform = None

    ):
        data = np.load(npz_file)
        self.data = data["images"]
        self.targets = data["labels"]
        self.multiclass_targets = self.targets
        self.data, self.y_true, self.y = binarize_dataset(
            features = self.data,
            targets = self.targets,
            pos_class = pos_class,
            neg_class = neg_class,
            num_labeled = num_labeled,
            num_unlabeled = num_unlabeled,
            prior = tgt_prior
        )
        self.transform = transform
        # self.data = torch.tensor(self.data, dtype=torch.float32).permute(0, 3, 1, 2)
        # self.targets = torch.tensor(self.targets, dtype=torch.long)
    def __len__(self):
         return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        y_true = self.y_true[idx]
        y = self.y[idx]

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)
        # else:
        #     img = transforms.ToTensor()(img)

        return img, y_true, y 
         


class BinaryCIFAR10C(Dataset):
    def __init__(
        self,
        pos_class: List,
        neg_class: List = None,
        root=root_dir,
        train: bool = True,
        num_labeled: int = None,
        num_unlabeled: int = None,
        tgt_prior: float = 0.5, 
        transform = None

    ):
        self.data, self.targets = self.load_cifar10c()
        
        self.data, self.y_true, self.y = binarize_dataset(
            features = self.data,
            targets = self.targets,
            pos_class = pos_class,
            neg_class = neg_class,
            num_labeled = num_labeled,
            num_unlabeled = num_unlabeled,
            prior = tgt_prior,
        )
        self.transform = transform
    def load_cifar10c(self):
        file_list = [
            "brightness.npy", "contrast.npy", "defocus_blur.npy", "elastic_transform.npy",
            "fog.npy", "frost.npy", "gaussian_blur.npy", "gaussian_noise.npy", 
            "glass_blur.npy", "impulse_noise.npy", "jpeg_compression.npy",
            "motion_blur.npy", "pixelate.npy", "saturate.npy", "shot_noise.npy",
            "snow.npy", "spatter.npy", "speckle_noise.npy", "zoom_blur.npy"
        ]

        images = []
        labels = []
        labels_path = "./data/CIFAR-10-C/labels.npy"
        all_labels = np.load(labels_path)

        for file in file_list:
            file_path = os.path.join("./data/CIFAR-10-C/", file)
            img_data = np.load(file_path) 
            indices = np.random.choice(len(img_data), size=550, replace=False)

            sampled_images = img_data[indices]
            sampled_labels = all_labels[indices]

            images.append(sampled_images)
            labels.append(sampled_labels)
       
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0) 
        return images, labels

    def __len__(self):
         return len(self.data)
    

    def __getitem__(self, idx):
         img, y_true, y = self.data[idx], self.y_true[idx], self.y[idx]
        
         img = to_pil_image(img)
         if self.transform is not None:
             img = self.transform(img)
         return img, y_true, y 


class BinaryCINICIMGNET(Dataset):
    def __init__(
        self,
        pos_class: List,
        neg_class: List = None,
        root=root_dir,
        train: bool = True,
        num_labeled: int = None,
        num_unlabeled: int = None,
        tgt_prior: float = 0.5,
        transform = None

    ):
        data = datasets.ImageFolder('./data/cinic-10-imagenet/train',
                    transform=transforms.Compose([transforms.ToTensor()]))
        X_list = []
        y_list = []
        
        for img, label in data:
            img_np = img.numpy()                      # [0.0, 1.0], float32, shape: (C, H, W)
            img_np = (img_np * 255).clip(0, 255)      # [0.0, 255.0], float32
            img_np = img_np.astype(np.uint8)          # [0, 255], uint8
            X_list.append(img_np)
            y_list.append(label)

        self.data = np.transpose(np.stack(X_list), (0, 2, 3, 1))             # shape: (N, H, W, C)
        self.targets = np.array(y_list)
        self.multiclass_targets = self.targets
        self.data, self.y_true, self.y = binarize_dataset(
            features = self.data,
            targets = self.targets,
            pos_class = pos_class,
            neg_class = neg_class,
            num_labeled = num_labeled,
            num_unlabeled = num_unlabeled,
            prior = tgt_prior
        )
        self.transform = transform
        # self.data = torch.tensor(self.data, dtype=torch.float32).permute(0, 3, 1, 2)
        # self.targets = torch.tensor(self.targets, dtype=torch.long)
    def __len__(self):
         return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        y_true = self.y_true[idx]
        y = self.y[idx]

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)
        # else:
        #     img = transforms.ToTensor()(img)

        return img, y_true, y 

 

def binarize_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    pos_class: List,
    neg_class: List=None,
    num_labeled: int=0,
    num_unlabeled: int=None,
    prior: float=None
):
    # Positive and Negative data indices
    p_data_idx = np.where(np.isin(targets, pos_class))[0]
    n_data_idx = np.where(np.isin(targets, neg_class) if neg_class else
                          np.isin(targets, pos_class, invert=True))[0]

    # Obtain labeled positive data 
    num_pos_labeled_per_cls = int(num_labeled / len(pos_class))
    p_ix = []
    for cls in pos_class:
        p_cls = np.where(np.isin(targets, cls))[0]
        p_ix.extend(np.random.choice(a=p_cls, size=num_pos_labeled_per_cls, replace=False))
    p_data = features[p_ix]
    p_labels = np.ones(len(p_data), dtype=targets.dtype)

    # Obtain unlabeled data
    if num_unlabeled and prior:
        n_pu = int(prior * num_unlabeled)
        n_nu = num_unlabeled - n_pu
        pu_ix = np.random.choice(a=p_data_idx, size=n_pu,
                                 replace=False if n_pu <= len(p_data_idx) else True)
        nu_ix = np.random.choice(a=n_data_idx, size=n_nu,
                                 replace=False if n_nu <= len(n_data_idx) else True)
        u_ix = np.concatenate((pu_ix, nu_ix), axis=0)
        u_data = features[u_ix]

        y_true = np.concatenate((np.ones(len(pu_ix), dtype=int), -np.ones(len(nu_ix), dtype=int)))
        y = np.zeros(len(u_data), dtype=int)
    else:
        remaining_p_ix = np.setdiff1d(p_data_idx, p_ix)
        u_ix = np.concatenate((remaining_p_ix, n_data_idx), axis=0)
        u_data = features[u_ix]

        y_true = np.concatenate((np.ones(len(remaining_p_ix), dtype=int), -np.ones(len(n_data_idx), dtype=int)))
        y = np.zeros(len(u_data), dtype=int)



    # Create final PU dataset
    features = np.concatenate((p_data, u_data), axis=0)
    y_true = np.concatenate((np.ones(len(p_data), dtype=int), y_true), axis=0)
    y = np.concatenate((p_labels, y), axis=0)

    # Shuffle and return
    features, y_true, y = shuffle(features, y_true, y, random_state=0)
    return features, y_true, y
    



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


class TransformCL:
    def __init__(self, dataset):
        
        if dataset in ['mnist', 'usps', 'svhn']:
            self.t1 = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop(28, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=mnist_avg, std=mnist_std)
            ])
            self.t2 = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop(28, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mnist_avg, std=mnist_std)
            ])

        elif dataset in ['cifar', 'cifarv2', 'cifar10c', 'cinic']:
            self.t1 = transforms.Compose([
                transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar_avg, std=cifar_std)
            ])
            self.t2 = transforms.Compose([
                transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=cifar_avg, std=cifar_std)
            ])

    def __call__(self, x):
        return self.t1(x), self.t2(x)



class TwoViewDataset(Dataset):
    def __init__(self, base, pair_transform: TransformCL):
        self.base = base
        self.pair_t = pair_transform

    def __len__(self): 
        return len(self.base)
    
    def __getitem__(self, idx):
        item = self.base[idx]
        img = item[0]
        
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)

        y1, y2 = self.pair_t(img)
        label = item[1] if len(item) > 1 else -1
        return (y1, y2), label, idx
    

def get_stats(dataset_name: str):
    """return (mean, std, crop_size)"""
    if dataset_name in ['cifar', 'cifarv2', 'cifar10c', 'cinic']:
        return tuple(cifar_avg), tuple(cifar_std), 32
    elif dataset_name in ['mnist', 'usps', 'svhn']:
        return tuple(mnist_avg), tuple(mnist_std), 28
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")



class TransformDataset(Dataset):
    def __init__(self, base, x_transform):
        self.base = base
        self.x_transform = x_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y_true, y = self.base[idx]
        x = self.x_transform(x)
        return x, y_true, y


class TransformFixMatch(object):
    def __init__(self, dataset_name, mean, std, crop_size):
        if dataset_name in ['cifar', 'cifarv2', 'cifar10c', 'cinic']:
            self.weak = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.Resize((crop_size, crop_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((crop_size, crop_size), antialias=True),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif dataset_name in ['mnist', 'usps', 'svhn']:
            self.weak = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(28, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(crop_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(28, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                RandAugmentMC(n=2, m=10),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            raise ValueError(f"Unsupported dataset for FixMatch: {dataset_name}")

        if dataset_name in ['mnist', 'usps', 'svhn']:
            self.normalize = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(crop_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.normalize = transforms.Compose([
                transforms.Resize(crop_size, interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def weak_only(self):
        return self.weak

    def strong_only(self):
        return self.strong

    def eval_only(self):
        return self.normalize
    
    def pair(self, x):
        w = self.weak(x)
        s = self.strong(x)
        return w, s

    def __call__(self, x):

        return self.pair(x)



def make_dataset(dataset, dataset_name: str, batch_size: int, role: str):
    mean, std, crop_size = get_stats(dataset_name)
    st_transform = TransformFixMatch(dataset_name, mean, std, crop_size)
    ds = copy.deepcopy(dataset)

    base = ds.dataset if isinstance(ds, Subset) else ds
    if role == 'weak_train':
        base.transform = st_transform.weak_only()
      
    elif role in ('weak_val', 'test'):
        base.transform = st_transform.eval_only()
       
    elif role == 'tot_test':
        base.transform = st_transform  # __call__ → (weak, strong)
       
    else:
        raise ValueError(f"Unknown role: {role}")
    
    return ds

def dataset_to_tensors(ds, batch_size=1024, num_workers=2):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=num_workers, pin_memory=True)
    xs, yts, ys = [], [], []
    for x, y_true, y in loader:
        # x는 transforms.ToTensor() 이후 Tensor
        xs.append(x)
        yts.append(torch.as_tensor(y_true))
        ys.append(torch.as_tensor(y))
    X  = torch.cat(xs, dim=0)
    Yt = torch.cat(yts, dim=0).long()
    Y  = torch.cat(ys, dim=0).long()
    return TensorDataset(X, Yt, Y)
