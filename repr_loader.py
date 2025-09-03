import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

from randaugment import RandAugmentMC


def make_loader(dataset, dataset_name, train=False, test=True, batch_size=128):
    if dataset_name in ['cifar','cifarv2', 'cifar10c']: 
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        crop_size=32
    elif dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        crop_size=20
    elif dataset_name == 'usps': 
        mean, std = (0.1703,), (0.3946,) 
        crop_size=20  
    elif dataset_name == 'svhn': 
        mean, std = (0.1307,), (0.3081,)
        crop_size=20
    elif dataset_name =='cinic':
        mean, std = (0.4752, 0.4694, 0.4258), (0.2405, 0.2366, 0.2578)
        crop_size=32

    # weak augmentation
    transform_labeled_cifar = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    

    transform_labeled_mnist = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    transform_val_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    
    transform_val_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    

    if dataset_name == 'cifar':          # 32×32 입력
        if train:                        # source‑train
            dataset.transform = transform_labeled_cifar
        elif test:                       # val / test
            dataset.transform = transform_val_cifar
        else:                            # target‑U  (weak,strong)
            dataset.transform = TransformFixMatch(
                mean=mean, std=std, dataset_name=dataset_name, crop_size=crop_size)        # ★ s

    elif dataset_name in ['mnist', 'usps', 'svhn']:    
        if train:
            dataset.transform = transform_labeled_mnist
        elif test:
            dataset.transform = transform_val_mnist
        else:
            dataset.transform = TransformFixMatch(
                mean=mean, std=std, crop_size=crop_size)    
                
    else:                               
        if test:
            dataset.transform = transform_val_cifar
        else:                             # target‑U
            dataset.transform = TransformFixMatch(mean=mean, std=std, dataset_name=dataset_name, crop_size=crop_size)

            
    if train:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True 
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False 
        )
    return loader




def make_representation(encoder, loader, with_head, train=False, val=False, batch_size=128):

    encoder.cuda().eval();
    all_y_true = []
    all_y = []
    if train == True:
        all_repr = []
        with torch.no_grad():
            for img, y_true, y in loader:
                img = img.cuda(non_blocking=True)
                if with_head == True:
                    z, _ = encoder(img)
                else:
                    z= encoder.forward_backbone(img)
                
                all_repr.append(z.cpu())
                all_y_true.append(y_true)
                all_y.append(y)
    
        all_repr = torch.cat(all_repr, dim=0)
        all_y_true = torch.cat(all_y_true, dim=0)
        all_y = torch.cat(all_y, dim=0)
        repr_dataset = TensorDataset(all_repr, all_y_true, all_y)
        repr_loader = DataLoader(
            repr_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True 
        )

        # pos: y == 1
        pos_mask = (all_y == 1)
        repr_pos_dataset = TensorDataset(all_repr[pos_mask], all_y_true[pos_mask], all_y[pos_mask])

        # unl: y == -1
        unl_mask = (all_y == -1)
        repr_unl_dataset = TensorDataset(all_repr[unl_mask], all_y_true[unl_mask], all_y[unl_mask])

        if not val:
            return repr_loader, repr_pos_dataset, repr_unl_dataset
        else:
            return repr_dataset


    elif train == False:
        all_strong_repr = []
        all_weak_repr = []
        with torch.no_grad():
            for (weak_img, strong_img), y_true, y in loader:
                y_true, y = y_true.cuda(), y.cuda()
                weak_img=weak_img.cuda(non_blocking=True)
                strong_img=strong_img.cuda(non_blocking=True)
                if with_head == True:
                    weak_z, _ = encoder(weak_img)
                    strong_z, _ = encoder(strong_img)
                else:
                    weak_z = encoder.forward_backbone(weak_img)
                    strong_z = encoder.forward_backbone(strong_img)
                
                
                all_weak_repr.append(weak_z.cpu())
                all_strong_repr.append(strong_z.cpu())
                all_y_true.append(y_true)
                all_y.append(y)
    
        all_weak_repr = torch.cat(all_weak_repr, dim=0)
        all_strong_repr = torch.cat(all_strong_repr, dim=0)

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y = torch.cat(all_y, dim=0)

        repr_dataset = CustomDataset(
            all_weak_repr,
            all_strong_repr,
            all_y_true,
            all_y
            )
        return repr_dataset

    



class TransformFixMatch(object):
    def __init__(self, mean, std, dataset_name, crop_size):
        if dataset_name in ['cifar', 'cifarv2', 'cifar10c', 'cinic']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=crop_size,
                                    padding=int(crop_size*0.125),
                                    padding_mode='reflect')])
            
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=crop_size,
                                    padding=int(crop_size*0.125),
                                    padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
            


        elif dataset_name in ['mnist', 'usps', 'svhn']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=crop_size,
                                    padding=int(crop_size*0.125),
                                    padding_mode='reflect'),
                transforms.Grayscale(num_output_channels=1)])
            
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=crop_size,
                                    padding=int(crop_size*0.125),
                                    padding_mode='reflect'),
                RandAugmentMC(n=2, m=10),
                transforms.Grayscale(num_output_channels=1)])
            
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])


    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    


class CustomDataset(Dataset):
    def __init__(self, weak_z, strong_z, y_true, y):
        assert len(weak_z) == len(strong_z) == len(y_true) == len(y)
        self.w = weak_z      
        self.s = strong_z
        self.yt = y_true    
        self.y  = y

    def __len__(self):
        return len(self.w)

    def __getitem__(self, idx):
        z_pair   = (self.w[idx], self.s[idx])  
        label_pair = (self.yt[idx], self.y[idx])  
        return z_pair, self.yt[idx], self.y[idx]

