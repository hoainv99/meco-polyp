import os
import cv2
import argparse
import numpy as np
import  torch.nn as nn
from pydantic import DictError
from sklearn.metrics import confusion_matrix  
from torchmetrics import IoU
import copy
# import torchvision.transforms.functional as transforms_f
import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
# from mydataloader import transform
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms.functional as transforms_f
from PIL import Image
import torchvision.transforms as transforms
import random
from PIL import Image

def _dequeue_and_enqueue(model, reps, labels, masks, probs):

    batch_size = reps.shape[0]
    feat_dim = reps.shape[1]
    prob_dim = probs.shape[1]
    valid_pixel = labels * masks 
    reps = reps.permute(0, 2, 3, 1)
    memory_size = 10000
    memory_size_hard  = 10000
    pixel_update_freq = 500
    pixel_hard_update_freq = 500

    num_segments = torch.unique(labels)
    for bz in range(batch_size):
        this_rep = reps[bz]

        for lb in range(3):

            valid_pixel_seg = valid_pixel[bz, lb,:,:]  # select binary mask for i-th class
            
            if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
                continue

            prob_seg = probs[bz, lb,:,:]

            rep_mask_all = (prob_seg < 0.99)*valid_pixel_seg.bool() 
            rep_all = this_rep[rep_mask_all]
            rep_mask_hard = (prob_seg < 0.9) * valid_pixel_seg.bool()  # select hard queries
            rep_hard = this_rep[rep_mask_hard]

            # else:
            # rep_all = this_rep[valid_pixel_seg.bool() ]
            # rep_mask_hard = (prob_seg < 0.95) * valid_pixel_seg.bool()  # select hard queries
            # rep_hard = this_rep[rep_mask_hard]

            # rep_hard =  rep[rep_mask_hard]
            num_pixel_all = rep_all.shape[0]
            perm_all = torch.randperm(num_pixel_all)
            K_all = min(num_pixel_all, pixel_update_freq)
            this_rep_all = rep_all[perm_all[:K_all], :]
            ptr = int(model.rep_all_ptr[lb])

            if ptr + K_all >= memory_size:
                model.rep_all_queue[lb,-K_all:, :] = this_rep_all
                model.rep_all_ptr[lb] = 0
            else:
                model.rep_all_queue[lb,ptr:ptr + K_all, :] = this_rep_all
                model.rep_all_ptr[lb] = model.rep_all_ptr[lb] + K_all

                # print(model.rep_all_ptr)
            num_pixel_hard = rep_hard.shape[0]
            perm_hard = torch.randperm(num_pixel_hard)
            K_hard = min(num_pixel_hard, pixel_hard_update_freq)
            this_rep_hard = rep_hard[perm_hard[:K_hard], :]
            ptr = int(model.rep_hard_ptr[lb])
            if ptr + K_hard >= memory_size_hard:
                model.rep_hard_queue[lb, -K_hard:, :] = this_rep_hard
                model.rep_hard_ptr[lb] = 0
            else:
                model.rep_hard_queue[lb, ptr:ptr + K_hard, :] = this_rep_hard
                model.rep_hard_ptr[lb] = model.rep_hard_ptr[lb] + K_hard
    
    return model 

def transform1(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True, training = False):
    # image = image.long()
    _,raw_h, raw_w = image.shape
    scale_ratio = random.uniform(scale_size[0], scale_size[1])
    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    # print(label.shape)
    image = transforms_f.resize(image, resized_size)
    label = transforms_f.resize(label, resized_size)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)
    
    if torch.rand(1) > 0.5 and augmentation:

        # image = image.long()
        # print(image.shape)
        # image_cv = image.permute(1,2,0).data.cpu().numpy().astype(np.float32)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2HSV)
        # H_ori = np.sum(image_cv[:,:,0])
        # print(np.max(H_ori))
        # print(np.min(H_ori))
        # S_ori = np.sum(image_cv[:,:,1])
        # V_ori = np.sum(image_cv[:,:,2])
        # image_show = image.permute(1,2,0).data.cpu().numpy()
        # # print(image_show)
        # cv2.imwrite("img_show1.jpg", image_show[:,:,::-1])
        # assert 1==0
        # Random color jitter
        # color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
        if torch.rand(1) > 0.5:
            color_transform = transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), 0., 0.)  
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            
            image = color_transform(image)
            # B = float(torch.empty(1).uniform_(0.75, 1.25))
            # image = transforms_f.adjust_brightness(image, B)
            # # image = image.long()
            # C = float(torch.empty(1).uniform_(0.75, 1.25))
            # image = transforms_f.adjust_contrast(image, C)
            # image = image.long()
        if torch.rand(1) > 0.5:
            color_transform = transforms.ColorJitter(0, 0, (0.8, 1.2), (-0.1,0.1))  
        #     # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)
            # S = float(torch.empty(1).uniform_(0.75, 1.25))
            # image = transforms_f.adjust_saturation(image, S)
            # image = image.long()
            # H = float(torch.empty(1).uniform_(-0.5, 0.5))
            # print(H)
            # image = transforms_f.adjust_hue(image, 1e-6)
            # image = image.long()

    #     # Random Gaussian filter
    #     # if torch.rand(1) > 0:
    #     #     sigma = random.uniform(1, 1.75)
    #     #     image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    #     # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)
        if torch.rand(1) > 0.5:
            image = transforms_f.vflip(image)
            label = transforms_f.vflip(label)
            if logits is not None:
                logits = transforms_f.vflip(logits)

        # image_show = image.permute(1,2,0).data.cpu().numpy().astype(np.float32)
        # # label_show =label*255
        # print(image_show)
        # # label_show = label_show.data.cpu().numpy().squeeze(0)
        # image_cv = image.permute(1,2,0).data.cpu().numpy().astype(np.float32)
        # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2HSV)
        # cv2.imwrite("img_show2.jpg", image_show[:,:,::-1])
        # H_aug = np.sum(image_cv[:,:,0])
        # S_aug = np.sum(image_cv[:,:,1])
        # V_aug = np.sum(image_cv[:,:,2])
        # print(H_aug/H_ori)
        # print(S_aug/S_ori)
        # print(V_aug/V_ori)
        # assert 1==0
    # image = torch.FloatTensor(image)
    # image = image  / 255.


    # image = image.permute(2,0,1)
    # image = torch.from_numpy(image)
    # label = torch.from_numpy(label)
    # logits = torch.from_numpy(logits)
    if logits is not None:
        return image, label, logits
    else:
        return image, label
# --------------------------------------------------------------------------------
# Define EMA: Mean Teacher Framework
# --------------------------------------------------------------------------------
class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1


# --------------------------------------------------------------------------------
# Define Polynomial Decaycompute_supervised_loss
# --------------------------------------------------------------------------------
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


# --------------------------------------------------------------------------------
# Define training losses
# --------------------------------------------------------------------------------
def compute_supervised_loss(pred, mask, reduction=True):
    # if reduction:
    # loss = F.cross_entropy(pred, mask, ignore_index=-1)
    # else:
    # loss = F.cross_entropy(predict, target, ignore_index=-1, reduction='none')
    # return loss
    # mask  = mask.squeeze(1).float()
    # print(mask)
    # mask = mask.squeeze(1).long()
    mask_onehot = label_onehot(mask, 2)
    # a = F.avg_pool2d(mask_onehot, kernel_size=51, stride=1, padding=25)- mask_onehot
    weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask_onehot, kernel_size=51, stride=1, padding=25) - mask_onehot
        )
    # weit

    # mask = mask.squeeze(1).long()
    wbce = F.cross_entropy(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
   
    pred = torch.softmax(pred, dim = 1)


    inter = ((pred * mask_onehot) * weit).sum(dim=(2, 3))
    union = ((pred + mask_onehot) * weit).sum(dim=(2, 3))
    # wiou = 1 - (inter_1 + 1) / (union_1 - inter_1 + 1)
    # wiou = 1 - (inter_1 + 1) / (union_1 - inter_1 + 1)
    wiou = 1 - (inter + 1) / (union - inter + 1)
    # inter_2 = ((pred_2 * mask) * weit).sum(dim=(2, 3))
    # union_2 = ((pred_2 + mask) * weit).sum(dim=(2, 3))
    # wiou_2 = 1 - (inter_2 + 1) / (union_2 - inter_2 + 1)
    return (wbce + wiou).mean()
def split_mask(neo_mask):
    polyp_mask = neo_mask[:, [0], :, :] + neo_mask[:, [1], :, :]
    # neo, non-neo and background
    neo_mask = neo_mask[:, [0, 1, 2], :, :]
    
    return polyp_mask, neo_mask
def structure_loss(pred, mask):
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
def structure_loss1(pred, mask):
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

    return wce + wiou


def compute_unsupervised_loss(predict, target, logits, strong_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels

    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    loss = structure_loss(predict, target)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss

def recall_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-07)
    return recall


def precision_m(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-07)
    return precision
    
def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-07))


def jaccard_m(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-07)
# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def compute_reco_loss(rep, label, mask, prob, rep_all_queue, rep_hard_queue, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h = rep.shape

    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
        rep_all =  rep_all_queue[i,:,:]
        rep_all = rep_all[(rep_all != 0).all(dim=-1),:]
        hard_list = rep_hard_queue[i,:,:]
        hard_list = hard_list[(hard_list != 0).all(dim=-1),:]
        if i!=0:
            x_all = abs(int(seg_feat_all_list[0].shape[0]) - int(rep[valid_pixel_seg.bool()].shape[0]))
            x_hard = abs(int(seg_feat_hard_list[0].shape[0]) - int(rep[rep_mask_hard].shape[0]))
            # print(x_all, x_hard)
            perm_all = torch.randperm(rep_all[i].shape[0])
            perm_hard = torch.randperm(hard_list[i].shape[0])
            rep_all  = torch.cat((rep[valid_pixel_seg.bool()],rep_all[perm_all[:x_all],:]), dim =0 )
            hard_list = torch.cat((rep[rep_mask_hard],hard_list[perm_hard[:x_hard],:]), dim =0 )

            seg_proto_list.append(torch.mean(rep_all, dim=0, keepdim=True))

            seg_feat_all_list.append(rep_all)
            seg_feat_hard_list.append(hard_list)
            seg_num_list.append(int(rep_all.shape[0]))
        else:
            seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
            seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
            seg_feat_hard_list.append(rep[rep_mask_hard])
            seg_num_list.append(int(valid_pixel_seg.sum().item()))
    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        # backgr_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):

            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])

                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)
                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            # if i ==0:
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
            # else:
            #     reco_loss = reco_loss + 0.4*F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))           
        return reco_loss/valid_seg
def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index

# --------------------------------------------------------------------------------
# Define evaluation metrics
# --------------------------------------------------------------------------------
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item(), acc.item()


# --------------------------------------------------------------------------------
# Define useful functions
# --------------------------------------------------------------------------------
def label_binariser(inputs):
    outputs = torch.zeros_like(inputs).to(inputs.device)
    index = torch.max(inputs, dim=1)[1]
    outputs = outputs.scatter_(1, index.unsqueeze(1), 1.0)
    return outputs


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def denormalise(x, imagenet=True):
    if imagenet:
        # x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        # x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        x = x / 255.
        return x
    else:
        return (x + 1) / 2


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)




class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')
        self.crit = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        # ce_loss = self.crit(logits, label.float())
        ce_loss = self.crit(logits, label.long())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

    # mask = mask.float()
    # weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # wfocal = FocalLossV1()(pred, mask)
    # wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    # inter = ((pred * mask)*weit).sum(dim=(2, 3))
    # union = ((pred + mask)*weit).sum(dim=(2, 3))
    # wiou = 1 - (inter + 1)/(union - inter+1)
    # return (wfocal + wiou).mean()
# --------------------------------------------------------------------------------
# Define semi-supervised methods (based on data augmentation)
# --------------------------------------------------------------------------------
def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_unsup_data(data, target, logits, mode='cutout'):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutout':
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            target[i][(1 - mix_mask).bool()] = -1

            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(target[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long(), new_logits



class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1
import time
def tensor_to_pil(im, label, logits):

    # im = im * 255.
    # im = im.data.cpu().numpy()
    label = label.unsqueeze(0).float()
    # label = label.data.cpu().numpy()
    logits = logits.unsqueeze(0)
    # logits = logits.data.cpu().numpy()
    return im, label, logits
def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform1(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list), torch.cat(label_list), torch.cat(logits_list)
    return data_trans.to(device), label_trans.to(device), logits_trans.to(device)

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    # y_pred = y_pred.flatten()
    # y_true = y_true.flatten()
    # current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # # compute mean iou
    # intersection = np.diag(current)
    # ground_truth_set = current.sum(axis=1)
    # predicted_set = current.sum(axis=0)
    # union = ground_truth_set + predicted_set - intersection
    # IoU = intersection / union.astype(np.float32)
    # return np.mean(IoU)
    iou = IoU(num_classes=2)
    return iou(y_pred, y_true)

def compute_dice(y_true, y_pred, smooth=1):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice.item()
