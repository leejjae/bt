from typing import Tuple, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# -------------------------
# 1) 28×28 1ch backbone
# -------------------------


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(16, 120, kernel_size=5),
        #     nn.ReLU()
        # )
        
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(84, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(32,1)
    
    @property
    def feature_dim(self):
        return 400
    
    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward_features(self, x):
        return self.encode(x)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        




class CNNcifar(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64, dropout: float = 0.0):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4

        # Stem (32x32 유지)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),

            nn.Conv2d(w1, w1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )

        # Stage 2 (32 -> 16)
        self.stage2 = nn.Sequential(
            nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),

            nn.Conv2d(w2, w2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
        )

        # Stage 3 (16 -> 8)
        self.stage3 = nn.Sequential(
            nn.Conv2d(w2, w3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),

            nn.Conv2d(w3, w3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(w3, 1, bias=True)

        # 기존 초기화 헬퍼 사용 (model.py 내 정의)
        self.apply(kaiming_init)  # :contentReference[oaicite:1]{index=1}

    @property
    def feature_dim(self):
        return self.head.in_features
    

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                 # (N, w1, 32, 32)
        x = self.stage2(x)               # (N, w2, 16, 16)
        x = self.stage3(x)               # (N, w3, 8, 8)
        x = F.adaptive_avg_pool2d(x, 1)  # (N, w3, 1, 1)
        x = torch.flatten(x, 1)          # (N, w3)
        return x         # (N,)
    
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.stem(x)                 # (N, w1, 32, 32)
    #     x = self.stage2(x)               # (N, w2, 16, 16)
    #     x = self.stage3(x)               # (N, w3, 8, 8)
    #     x = F.adaptive_avg_pool2d(x, 1)  # (N, w3, 1, 1)
    #     x = torch.flatten(x, 1)          # (N, w3)
    #     x = self.dropout(x)
    #     logit = self.head(x)             # (N, 1)
    #     return logit.squeeze(1)          # (N,)





# # ==== CIFAR ResNet-18 / -50 (예: CIFARResNet) ====
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1):
#         super().__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.downsample = None
#         if stride != 1 or in_planes != planes:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         identity = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out = self.relu(out + identity)
#         return out



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)


        feat_dim = 512*block.expansion
        hidden_dim = int(feat_dim/4)
        self.head1 = nn.Linear(feat_dim, hidden_dim, bias=True)
        self.head2 = nn.Linear(hidden_dim, 1, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.apply(kaiming_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    @property
    def feature_dim(self):
        return self.head1.in_features
    
    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.head1(x)
        x = self.bn2(x)
        x = self.relu(x)
        logit = self.head2(x)          # (N, 1)
        return logit.squeeze(1)       # (N,)






def cifar_resnet18(num_classes=1): 
    return CifarResNet(BasicBlock, [2,2,2,2], num_classes)


def cifar_resnet34(num_classes=1):  
    return CifarResNet(BasicBlock, [3,4,6,3], num_classes)


def cifar_resnet50(num_classes=1):  
    return CifarResNet(Bottleneck, [3,4,6,3], num_classes)


def cifar_resnet101(num_classes=1): 
    return CifarResNet(Bottleneck, [3,4,23,3], num_classes)


def cifar_resnet152(num_classes=1): 
    return CifarResNet(Bottleneck, [3,8,36,3], num_classes)
# -------------------------
# Factory
# -------------------------



def build_model(arch, **kwargs):
    if arch == "lenet":
        return LeNet(**kwargs)

    elif arch == 'cnn_cifar':
        return CNNcifar(**kwargs)
    
    elif arch == "cifar_resnet18":
        return CifarResNet(BasicBlock, [2,2,2,2])
    
    elif arch == "cifar_resnet50":
        return CifarResNet(BasicBlock, [3,4,6,3])

    else:
        raise ValueError(f"Unknown arch '{arch}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
