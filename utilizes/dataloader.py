import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, RandomSampler
from utilizes.augment import NoAugmenter, Augmenter
from torch import distributed as dist
import albumentations as A
import warnings
import cv2
import torch
warnings.filterwarnings('ignore')


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths, gt_paths, img_size, transforms=None, mode='train'):
        self.img_size = img_size
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.images = sorted(self.image_paths)
        self.gts = sorted(self.gt_paths)
        self.filter_files()
        self.size = len(self.images)
        self.transforms = transforms
        self.mode = mode
        self.img_size = img_size
        
    
    def __getitem__(self, idx):
        image_paths = self.image_paths[idx]
        gt_paths = self.gt_paths[idx]
        image_ = np.array(Image.open(image_paths).convert("RGB"))
        mask = np.array(Image.open(gt_paths).convert("L"))
        
        augmented = self.transforms(image=image_, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        mask_resize = mask
        # if os.path.splitext(os.path.basename(img_path))[0].isnumeric():
        mask = mask / 255

        # if self.mode == "train":
        #     mask = cv2.resize(mask, (self.img_size, self.img_size))
        # elif self.mode == "val":
        #     mask_resize = cv2.resize(mask, (self.img_size, self.img_size),interpolation = cv2.INTER_NEAREST)
        #     mask_resize = mask_resize[:, :, np.newaxis]

        #     mask_resize = mask_resize.astype("float32")
        #     mask_resize = mask_resize.transpose((2, 0, 1))

        # image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]

        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))

        if self.mode == "train":
            return np.asarray(image), np.asarray(mask)

        elif self.mode == "test":
            return (
                np.asarray(image),
                np.asarray(mask),
                os.path.basename(image_paths),
                np.asarray(image_),
            )
        else:
            return (
                np.asarray(image),
                np.asarray(mask),
                np.asarray(mask_resize),
            )


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
            img = np.array(img)
            return img

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("L")
            img = np.array(img)
            return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.img_size or w < self.img_size:
            h = max(h, self.img_size)
            w = max(w, self.img_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

class NeoDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths, gt_paths, img_size, transforms=None, mode='train'):
        self.img_size = img_size
        self.image_paths = image_paths
        self.gt_paths = gt_paths
        self.images = sorted(self.image_paths)
        self.gts = sorted(self.gt_paths)
        self.filter_files()
        self.size = len(self.images)
        self.transforms = transforms
        self.mode = mode
        self.img_size = img_size
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        gt_path = self.gts[idx]
        image_ = np.array(cv2.imread(image_path)[:,:,::-1])
        mask = np.array(cv2.imread(gt_path))

        mask[mask >= 128] = 255
        mask[mask < 128] = 0
        
        neo_gt = np.all(mask == [0, 0, 255], axis=-1).astype('float')
        non_gt = np.all(mask == [0, 255, 0], axis=-1).astype('float')

        augmented = self.transforms(image=image_, neo=neo_gt, non=non_gt)
        
        image, neo, non = augmented["image"], augmented["neo"], augmented['non']

        
        mask = np.stack([neo, non], axis=-1)
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate([mask, background], axis=-1)
        mask_resize = mask
        
        if self.mode == "train":
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        elif self.mode == "val":
            mask_resize = cv2.resize(mask, (self.img_size, self.img_size),interpolation = cv2.INTER_NEAREST)

            mask_resize = mask_resize.astype("float32")
            mask_resize = mask_resize.transpose((2, 0, 1))

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask.astype("float32")
        mask = mask.transpose((2, 0, 1))

        if self.mode == "train":
            return np.asarray(image), np.asarray(mask)

        elif self.mode == "test":
            return (
                np.asarray(image),
                np.asarray(mask),
                os.path.basename(image_path),
                np.asarray(image_),
            )
        else:
            return (
                np.asarray(image),
                np.asarray(mask),
                np.asarray(mask_resize),
            )


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
            img = np.array(img)
            return img

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("L")
            img = np.array(img)
            return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.img_size or w < self.img_size:
            h = max(h, self.img_size)
            w = max(w, self.img_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(
    image_paths,
    gt_paths,
    transforms,
    transforms_weak,
    batchsize,
    img_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    mode='train',
    use_ddp=False
):
    dataset = NeoDataset(image_paths, gt_paths, img_size, transforms=transforms, mode=mode)
    dataset2 = NeoDataset(image_paths, gt_paths, img_size, transforms=transforms_weak, mode=mode)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    data_loader2 = data.DataLoader(
        dataset=dataset2,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader,data_loader2

def get_test_loader(
    image_paths,
    gt_paths,
    transforms,
    batchsize,
    img_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    mode='train',
    use_ddp=False
):
    dataset = NeoDataset(image_paths, gt_paths, img_size, transforms=transforms, mode=mode)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader

if __name__ == '__main__':
    image_root = '/home/s/hungpv/polyps/datatest/test/images'
    gt_root = '/home/s/hungpv/polyps/datatest/test/label_images'
    
    image_paths = [os.path.join(image_root, i) for i in os.listdir(image_root)]
    gt_paths = [os.path.join(gt_root, i) for i in os.listdir(gt_root)]
    augment = Augmenter(prob=1)
    dataset = NeoDataset(image_paths, gt_paths, img_size=352, transforms=augment, mode='train')
    img, gt = dataset.__getitem__(3)
 
    img = img.transpose(1, 2, 0)
    cv2.imwrite('img.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255)
    gt = gt.transpose(1, 2, 0)
    neo, non, bgr = gt[:,:,0], gt[:,:,1], gt[:,:,2]
    cv2.imwrite('neo.jpg', neo * 255)
    cv2.imwrite('non.jpg', non * 255)
    cv2.imwrite('back.jpg', bgr * 255)
    cv2.imwrite('mask.jpg', neo * 255 + non * 255)
    # dataloader = get_loader(image_paths, gt_paths, transforms=augment, batchsize=2, img_size=352)
    # for i, (imgs, gts) in enumerate(dataloader):
    
    #     if i == 3:
    #         break
    

