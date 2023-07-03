from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F
from .mcc_loss import MCC_Loss
from .tversky_loss import TverskyLoss
from .ssim import SSIM

def split_mask(neo_mask):
    polyp_mask = neo_mask[:, [0], :, :] + neo_mask[:, [1], :, :]
    # neo, non-neo and background
    neo_mask = neo_mask[:, [0, 1, 2], :, :]
    
    return polyp_mask, neo_mask

def FocalTverskyLoss(inputs, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=1, ignore=None):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
    else:
        inputs = torch.softmax(inputs, dim=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp
    
    ft_score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ft_loss = (1 - ft_score) ** gamma
    
    return ft_loss.mean()

class multi_structure_loss(_Loss):
    def __init__(self):
        super(multi_structure_loss, self).__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        polyp_mask, neo_mask = split_mask(mask)

        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(neo_mask, kernel_size=51, stride=1, padding=25) - neo_mask
        )
        wce = F.cross_entropy(pred, torch.argmax(neo_mask, axis=1), reduction='none').mean()
        wce = (weit * wce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        
        pred = torch.softmax(pred, dim=1)
        inter = ((pred * neo_mask) * weit).sum(dim=(2, 3))
        union = ((pred + neo_mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wce + wiou).mean()

