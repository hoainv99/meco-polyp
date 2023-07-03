import torch

def CELoss(inputs, targets, ignore=None):
    if inputs.shape[1] == 1:
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
        ce_loss = torch.nn.functional.cross_entropy(inputs, torch.argmax(targets, axis=1), reduction='none')

    if ignore is not None:
        ignore = 1 - ignore.squeeze()
        ce_loss = ce_loss * ignore

    return ce_loss.mean()