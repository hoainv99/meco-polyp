from argparse import ArgumentParser
from logging import Logger
from multiprocessing.spawn import import_main_path
from numpy import False_
from utilizes.config import load_cfg
from utilizes.dataloader import get_loader
from utilizes.augment import Augmenter
import tqdm   
import cv2
import torch
from loguru import logger
from models.branch_model import CustomModel
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
        "-c", "--config", required=False, default="configs/neo_large.yaml"
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
        test_img_paths.extend(glob(os.path.join(i, "images", "*")))
        test_mask_paths.extend(glob(os.path.join(i, "label_images", "*")))

    test_img_paths.sort()
    test_mask_paths.sort()

    test_augprams = config["test"]["augment"]
    test_transform = Augmenter(**test_augprams, img_size=config['test']['dataloader']['img_size'])
    
    test_loader = get_loader(
        test_img_paths,
        test_mask_paths,
        transforms=test_transform,
        **config["test"]["dataloader"],
        mode="test",
    )

    test_size = len(test_loader)
    
    logger.info('Evaluating with test size {}'.format(test_size))
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

    tp_all = 0
    fp_all = 0
    fn_all = 0
    tn_all = 0

    mean_iou = 0
    mean_dice = 0

    mean_dice_neo = 0
    mean_dice_non = 0
    mean_iou_neo = 0
    mean_iou_non = 0

    mean_macro_dice_polyp = 0
    mean_macro_iou_polyp = 0
    mean_macro_dice_neo = 0
    mean_macro_iou_neo = 0
    mean_macro_dice_non = 0
    mean_macro_iou_non = 0

    mean_micro_dice_polyp = 0
    mean_micro_iou_polyp = 0
    mean_micro_dice_neo = 0
    mean_micro_iou_neo = 0
    mean_micro_dice_non = 0
    mean_micro_iou_non = 0



    if config['test']['visualize'] == True:
        logger.info("Eval with generate predict masks")
        visualize_dir = os.path.join(config['test']['visualize_dir'], name_scenario)
        os.makedirs(visualize_dir, exist_ok=True)

    logger.info(f"Start testing {len(test_loader)} images in {dataset} dataset")
    model.eval()

    polyp_metric = MetricForNeoplasm("polyp")
    neo_metric = MetricForNeoplasm("neo")
    non_metric = MetricForNeoplasm("non")

    for i, pack in tqdm.tqdm(enumerate(test_loader)):
        
        image, gt, name, _ = pack
        image_id = name[0].split('.')[0]
        # name = os.path.splitext(filename[0])[0]
        # ext = os.path.splitext(filename[0])[1]
        gt = np.asarray(gt, np.float32).round()
        neo_gt = gt[:, 0,:,:]
        non_gt = gt[:, 1,:,:]

        polyp_gt = neo_gt + non_gt

        image = image.cuda()


        with torch.no_grad():
            result = model(image)
        res = torch.nn.functional.upsample(
            result, size=gt[0][0].shape, mode="bilinear", align_corners=False
        )

        res = res.softmax(dim=1)
        res = torch.argmax(
            res, dim=1, keepdims=True).squeeze().data.cpu().numpy()
        
        neo_pr = (res == 0).astype(np.float)
        non_pr = (res == 1).astype(np.float)

        # print(np.unique(non_pr))
        polyp_pr = neo_pr + non_pr

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

            saved_path = os.path.join(visualize_dir, '{}.png'.format(image_id))
            cv2.imwrite(saved_path, output)
        #     save_img(
        #         os.path.join(
        #             visualize_dir,
        #             "PR_" + model_name,
        #             "Hard",
        #             name + ext,
        #         ),
        #         res.round() * 255,
        #         "cv2",
        #         overwrite,
        #     )
        #     save_img(
        #         os.path.join(
        #             visualize_dir,
        #             "PR_" + model_name,
        #             "Soft",
        #             name + ext,
        #         ),
        #         res * 255,
        #         "cv2",
        #         overwrite,
        #     )
        #     mask_img = (
        #         np.asarray(img[0])
        #         + vis_x
        #         * np.array(
        #             (
        #                 np.zeros_like(res.round()),
        #                 res.round(),
        #                 np.zeros_like(res.round()),
        #             )
        #         ).transpose((1, 2, 0))
        #         + vis_x
        #         * np.array(
        #             (gt, np.zeros_like(gt), np.zeros_like(gt))
        #         ).transpose((1, 2, 0))
        #     )
        #     mask_img = mask_img[:, :, ::-1]
        #     save_img(
        #         os.path.join(
        #             visualize_dir,
        #             "GT_PR_" + model_name,
        #             name + ext,
        #         ),
        #         mask_img,
        #         "cv2",
        #         overwrite,
        #     )

        prs.append(non_pr)
        gts.append(non_gt)

        polyp_metric.cal(polyp_pr, polyp_gt)
        neo_metric.cal(neo_pr, neo_gt)
        non_metric.cal(non_pr, non_gt)

    polyp_metric.show()
    neo_metric.show()
    non_metric.show()

    return gts, prs


if __name__ == "__main__":
    main()
