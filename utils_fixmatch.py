import torch
from torch.nn import functional as F



def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

@torch.no_grad()
def make_soft_labels(logits_w, pos_thresh, neg_thresh, require_both=True):
    
    probs = torch.sigmoid(logits_w)           # 0‥1 확률
    mask_pos = probs >= pos_thresh
    mask_neg = probs <= neg_thresh
    mask = (mask_pos | mask_neg).float()      # BCE 계산할 위치

    if require_both and ((mask_pos.sum() == 0) or (mask_neg.sum() == 0)):
        # 둘 중 하나라도 없으면 이번 mini‑batch 전체 스킵
        dummy = torch.zeros_like(probs)
        return dummy, mask_pos, mask_neg, mask.float()

    # soft target = 확률 그대로
    return probs.detach(), mask_pos, mask_neg, mask.float()


# def make_labels_from_fixmatch(logits_w, tau=0.95, T=1.0, require_both=True):
#     p = torch.sigmoid(logits_w / T)           
#     conf = torch.maximum(p, 1.0 - p)           
#     mask = (conf >= tau).float()               
#     target = (p >= 0.5).float()               
#     if require_both:
#         mask_pos = (target == 1) & (mask == 1)
#         mask_neg = (target == 0) & (mask == 1)
#         if (mask_pos.sum() == 0) or (mask_neg.sum() == 0):
#             # pos/neg 둘 중 하나라도 없으면 dummy 리턴
#             dummy = torch.zeros_like(target)
#             return dummy, mask

#     return target.detach(), mask



def make_labels_from_fixmatch(logits_w, tau=0.95, T=1.0):
    # logits_w: (MuB,)
    p = torch.sigmoid(logits_w / T)
    conf = torch.maximum(p, 1.0 - p)
    mask = (conf >= tau).float()     # FixMatch mask
    target = (p >= 0.5).float()      # 0/1
    return target.detach(), mask