import argparse
import cv2
import os
import numpy as np
import torch
from PIL import Image
import glob
# from models.GCN_model import GCN_sim
# from models.vt_model_oneLSTM_sim import VideoCaption_sim
# # from models.vt_model_oneLSTM_attention_sim import VideoCaption_sim
# from models.vt_model_oneLSTM__ import VideoCaption
# from models.vt_model_oneLSTM_avg_ressim import VideoCaption_ressim
# from models.vt_model_oneLSTM_all import VideoCaption
import torchvision.transforms as transforms
import torch.nn.functional as F
import subprocess


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (bottom, left, top, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    bottom_line = max(rec1[1], rec2[1])
    top_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or bottom_line >= top_line:
        return 0.
    else:
        intersect = (right_line - left_line) * (top_line - bottom_line)
        return intersect / (sum_area - intersect)

def show(img, dets, idx):
    """Visual debugging of detections."""
    image_tmp = np.array(img)
    im = image_tmp[:,:,::-1].copy()
    dets=dets.cpu()
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, 1:])
        # print(bbox)
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (10, 10, 204), 2)
    return im

def concat_pic(cls_det):
    image = []
    im_list_2d = []
    rows = 4
    for idx in range(len(sample_frame)):
        image.append(show(imgs[idx], cls_det[idx], idx))
    for row in range(rows):
        im_list_2d.append(image[len(image) * row // rows: len(image) * (row+1) //rows])
    # im_list_2d = [image[:len(image)//2], image[len(image)//2:]]
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = transforms.Resize(size, interpolation)
        
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def process_align():
    cls_det_ = []
    cls_det_.append(cls_det[0][box].unsqueeze(0))

    sim = F.normalize(feat, p=2, dim=-1)
    sim = torch.matmul(sim, sim.transpose(0, 1))  # N, N
    sim = sim[:10] # 10, N

    box_n = cls_det[0] # 10, 5
    iou = torch.zeros_like(sim) # 10, N

    for i in range(1, 32):
        for j in range(10):
            for k in range(10):
                iou[j, i*10+k] = compute_iou(box_n[j], cls_det[i, k])

    sim += iou

    for i in range(1, 32):
        feat_single = feat[i*10 : (i+1)*10]  # box, d
        sim_single = sim[:, i*10 : (i+1)*10]  # box, box
        cls_single = cls_det[i]

        sim_ = sim_single[box]  # box
        _, idx = sim_.max(0)
        cls_det_.append(cls_single[idx].unsqueeze(0))
    cls_det_ = torch.stack(cls_det_)

    return cls_det_
    

transform = transforms.Compose([
                GroupResize(256),
                GroupCenterCrop(256),
                ])

dataset = 'MSRVTT'
# dataset = 'MSVD'

frames_path = './feats/{}/frames/'.format(dataset)
bbox_path = './feats/{}/bbox_conf_0.1_nms_0.5'.format(dataset)
feature_path = './feats/{}/uniform_batch_0.1/resnet_box'.format(dataset)


# videos = os.listdir(bbox_path)
# videos.sort()

video = 'video7184'
# video = 'QMJY29QMewQ_42_52'
box = 0


cls_dets = torch.load(bbox_path+'/'+video+'.pth')
feat = torch.load(feature_path+'/'+video+'.pth').cpu()  # N, d

imgs = glob.glob(frames_path + video +'/*.jpg')
imgs.sort()

sample_frame = np.linspace(0, len(imgs)-1, 32, dtype=np.int)
# sample_frame = sample_frame[:10]
feat_len = len(sample_frame)
imgs = [imgs[idx] for idx in sample_frame]


imgs = [Image.open(img).convert('RGB') for img in imgs]
imgs = transform(imgs)

cls_det = cls_dets[sample_frame]
for i in range(cls_det.shape[0]):
    cls_single = cls_det[i]
    cls_single[(cls_single[:,3]-cls_single[:,1]) * (cls_single[:,4]-cls_single[:,2])<=300] = 0
    score = cls_single[:, -1]
    _, order = torch.sort(score, 0, True)
    cls_det[i] = cls_single[order]

cls_det = cls_det[:, :10]  # 32, 10, 5

# cls_det = process_align()

before = concat_pic(cls_det)
before_save_path = 'pic/' + video +'new.jpg'
cv2.imwrite(before_save_path, before)
print(before_save_path)