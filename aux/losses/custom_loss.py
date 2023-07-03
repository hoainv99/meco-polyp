import torch.nn as nn
from torch.nn.modules.loss import _Loss

from .focal_tversky_loss import FocalTverskyLoss
from .CELoss import CELoss

def split_mask(neo_mask):

    polyp_mask = neo_mask[:, [0], :, :] + neo_mask[:, [1], :, :]
    neo_mask = neo_mask[:, [0, 1, 2], :, :]
    
    return polyp_mask, neo_mask

class CustomLoss(_Loss):
    __name__ = 'CustomLoss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, mask):
        polyp_mask, neo_mask = split_mask(mask)

        ce_loss = CELoss(y_pr, neo_mask, ignore=None)
        ft_loss = FocalTverskyLoss(y_pr, neo_mask, ignore=None)
        main_loss = ce_loss + ft_loss

        return main_loss