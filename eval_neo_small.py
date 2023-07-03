from argparse import ArgumentParser
from logging import Logger
from multiprocessing.spawn import import_main_path
from numpy import False_
from utilizes.config import load_cfg
from utilizes.dataloader import get_loader, get_test_loader
from utilizes.augment import Augmenter
import tqdm   
import cv2
import time
import torch
from loguru import logger
from models.branch_model import CustomModel
# from models.custom_model import CustomModel
import os
from glob import glob
from utilizes.visualize import save_img
from aux.metrics.metrics import *
import numpy as np
import torch.nn.functional as F
import imageio
from datetime import datetime


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=False, default="configs/neo_small.yaml"
    )
    args = parser.parse_args()

    logger.info("Loading config")
    config_path = args.config
    config = load_cfg(config_path)

    gts = []
    prs = []
    
    dataset = config["dataset"]["test_data_path"][0].split("/")[-1]
    
    test_img_paths = []
    test_mask_paths = []
    test_data_path = config["dataset"]["test_data_path"]
    for i in test_data_path:
        test_img_paths.extend(glob(os.path.join(i, "*")))
        test_mask_paths.extend(glob(os.path.join(i, "*")))

    test_img_paths.sort()
    test_mask_paths.sort()

    test_augprams = config["test"]["augment"]
    test_transform = Augmenter(**test_augprams, img_size=config['train']['dataloader']['img_size'])
    
    test_loader = get_test_loader(
        test_img_paths,
        test_mask_paths,
        transforms=test_transform,
        **config["test"]["dataloader"],
        mode="test",
    )

    test_size = len(test_loader)
    
    # logger.info('Evaluating with test size {}'.format(test_size))
    dev = config["test"]["dev"]
    logger.info("Loading model")
    model_prams = config["model"]
    backbone = config['model']['backbone']
    head = config['model']['head']
    model_name = f'{backbone}-{head}'
    
    if "save_dir" not in model_prams:
        save_dir = os.path.join("checkpoint/KCECE", model_name)
    else:
        save_dir = config["model"]["save_dir"]
    
    num_classes = config['model']['num_classes']
    try:
        model = CustomModel(backbone=str(backbone), decode=str(head), num_classes=num_classes, pretrained=False)
    except:
        logger.info('Can not load model :( try find out sth ...')
        
    device = torch.device(dev)
    if dev == "cpu":
        model.cpu()
    else:
        model.cuda()
    
    name_scenario = save_dir.split('/')[-1]
    checkpoint_names = os.listdir(save_dir)
    if config['test']['checkpoint_dir'] is not None:
        model_path = config['test']['checkpoint_dir']
        print(model_path)
    else:
        try:
            epoch_ckpts = [i.split('_')[-1].split('.')[0] for i in checkpoint_names]
            epoch_max = max(int(epoch_ckpts))
            model_path = os.path.join(save_dir, f'{model_name}_{epoch_max}.pth')

        except:
            model_path = os.path.join(save_dir, checkpoint_names[0])
    
    logger.info(f"Loading from {model_path}")
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device)["model_state_dict"]
        )
    except RuntimeError:
        model.load_state_dict(torch.load(model_path, map_location=device))

    if config['test']['visualize'] == True:
        logger.info("Eval with generate predict masks")
        visualize_dir = os.path.join(config['test']['visualize_dir'], name_scenario)
        os.makedirs(visualize_dir, exist_ok=True)

    logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")
    model.eval()
    start_time = time.time()
    print(visualize_dir)
    for i, pack in tqdm.tqdm(enumerate(test_loader)):
        
        image, gt, name, image_ = pack
        image_id = name[0].split('.')[0]
        b, h, w, c = image_.shape
        gt = np.asarray(gt, np.float32).round()
        neo_gt = gt[:, 0,:,:]
        non_gt = gt[:, 1,:,:]

        polyp_gt = neo_gt + non_gt

        image = image.cuda()

    
        with torch.no_grad():
            result,_ = model(image)

        print(result.shape)
        res = torch.nn.functional.upsample(
            result, size=gt[0][0].shape, mode="bilinear", align_corners=False
        )

        res = res.softmax(dim=1)
        res = torch.argmax(
            res, dim=1, keepdims=True).squeeze().data.cpu().numpy()
        
        neo_pr = (res == 0).astype(np.float)
        non_pr = (res == 1).astype(np.float)


        vis_x = config["test"]["vis_x"]
        if config["test"]["visualize"]:
            output = np.zeros(
            (res.shape[0], res.shape[1], 3)).astype(np.uint8)
            output[neo_pr > 0] = [0, 0, 255]
            output[non_pr > 0] = [0, 255, 0]
            output = np.zeros(
                (res.shape[-2], res.shape[-1], 3)).astype(np.uint8)
            output[(neo_pr > non_pr) * (neo_pr > 0.5)] = [0, 0, 255]
            output[(non_pr > neo_pr) * (non_pr > 0.5)] = [0, 255, 0]
            output = cv2.resize(output, (w, h))
            saved_path = os.path.join(visualize_dir, '{}.png'.format(image_id))
            cv2.imwrite(saved_path, output)
            
    end_time = time.time()
    fps = len(test_loader) / (end_time - start_time)
    logger.info(f'FPS: {fps}')
    return gts, prs


if __name__ == "__main__":
    main()
