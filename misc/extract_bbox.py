# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from torchvision.ops import nms 
import torchvision.transforms as transforms
from PIL import Image


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = '/home/mhy/detectron2/r101_model_final_f6e8b1.pkl'
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default="/home/mhy/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        # default="configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='frames',
        help="frames/clips",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='MSVD',
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    NMS = args.nms
    conf_thresh = args.confidence_threshold
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    print('threshold:', conf_thresh)
#==============================================================================
    dataset = args.dataset

    frames_path = '/data/video_captioning/feats/{}/frames/'.format(dataset)
    features_path =  '/data/video_captioning/feats/{}/bbox/'.format(dataset)
    features_nms_path =  '/data/video_captioning/feats/{}/bbox_conf_{}_nms_{}/'.format(dataset, conf_thresh, NMS)

    os.makedirs(features_path, exist_ok=True)
    os.makedirs(features_nms_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    videos = os.listdir(frames_path)
    videos.sort()

    for idx in tqdm.tqdm(range(len(videos))):
        video = videos[idx]
        feat_file = video+'.pth'
        if os.path.exists(features_path+'/'+feat_file):
            continue

        imgs = os.listdir(os.path.join(frames_path, video))
        imgs.sort()
        video_len = len(imgs)
        cls_dets = torch.zeros(video_len, 50, 5)
        cls_dets_keep = torch.zeros(video_len, 50, 5)

        if args.mode=='frames':
            frame_indices = np.linspace(0, video_len-1, 32, dtype=np.int)
        elif args.mode=='clips':
            frame_indices = np.linspace(0, video_len-32, 10, dtype=np.int)
        elif args.mode=='all':
            frame_indices = range(video_len)


        for i in frame_indices:
            path = os.path.join(frames_path, video, imgs[i])
            img = Image.open(path).convert('RGB')
            img = transform(img)
            img = transforms.ToPILImage()(img).convert('RGB')
            img = np.asarray(img)
            img = img[:, :, ::-1]

            # img = read_image(path, format="BGR")
            predictions, _ = demo.run_on_image(img)  # T, n, 5 score first
            boxes = predictions['instances'].pred_boxes.tensor
            scores = predictions['instances'].scores
            if scores.shape[0] == 0:
                continue
            cls_item = torch.cat((scores.view(scores.shape[0],-1), boxes), dim=1)
            box_num = cls_item.shape[0] if cls_item.shape[0] <= 50 else 50
            cls_dets[i][:box_num] = cls_item[:box_num]
                
            tmp = cls_item[cls_item[:, 0]>conf_thresh]
            boxes = []
            for j in range(len(tmp)):
                if (tmp[j, 3]-tmp[j,1]) * (tmp[j, 4]-tmp[j,2])>300:
                    boxes.append(tmp[j])
            if len(boxes)==0:
                continue
            cls_item = torch.stack(boxes)

            keep = nms(cls_item[:, 1:], cls_item[:, 0], NMS)
            box_num = len(keep) if len(keep) <= 50 else 50
            cls_item = cls_item[keep]
            cls_dets_keep[i][:box_num] = cls_item[:box_num]

        torch.save(cls_dets, features_path+'/'+feat_file)
        torch.save(cls_dets_keep, features_nms_path+'/'+feat_file)


