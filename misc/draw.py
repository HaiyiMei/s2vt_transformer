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
    for idx in range(len(sample_frame)):
        image.append(show(imgs[idx], cls_det[idx], idx))
    im_list_2d = [image[:len(image)//2], image[len(image)//2:]]
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

transform = transforms.Compose([
                GroupResize(256),
                GroupCenterCrop(256),
                ])

dataset = 'MSRVTT'
idx_v = 11
threshold = 0.5

frames_path = '../feats/{}/frames/'.format(dataset)
bbox_path = '../feats/{}/bbox_crop_256_2020'.format(dataset)
bbox_path = '../feats/{}/bbox_objectness'.format(dataset)


videos = os.listdir(bbox_path)
videos.sort()

video = videos[idx_v]
video = video[:-4]
# video = 'K-KVz3eqbnA_1_10'
video = 'ZN2_czSBSD0_240_250'
video = 'DIebwNHGjm8_27_38'
video = 'ACOmKiJDkA4_130_144'
video = 'video13'
print(video)

cls_dets = torch.load(bbox_path+'/'+video+'.pth')
# cls_dets = torch.load('/home/mhy/Documents/feats/MSVD/bbox_det/{}.pth'.format(video))

imgs = glob.glob(frames_path + video +'/*.jpg')
imgs.sort()

sample_frame = np.linspace(0, len(imgs)-1, 32, dtype=np.int)
sample_frame = sample_frame[16:]
feat_len = len(sample_frame)
imgs = [imgs[idx] for idx in sample_frame]

# subprocess.call('mkdir /home/mhy/Documents/detectron2/pic/{}'.format(video), shell=True)
# for img in imgs:
#     command = 'cp {} /home/mhy/Documents/detectron2/pic/{}'.format(img, video)
#     subprocess.call(command, shell=True)

imgs = [Image.open(img).convert('RGB') for img in imgs]
imgs = transform(imgs)

cls_det = cls_dets[sample_frame]
print(cls_det[0][:10])
print(cls_det.shape)

for i in range(cls_det.shape[0]):
    cls_det[i][cls_det[i][:, 0]<=threshold]=0
    cls_det[i][(cls_det[i][:, 3]-cls_det[i][:,1]) * (cls_det[i][:,4]-cls_det[i][:,2])<=300]=0

print(cls_det.shape)
print(cls_det[0][:10])

before = concat_pic(cls_det)
after = concat_pic(cls_det[:, :10])
print(bbox_path)
before_save_path = 'pic_demo/' + video + '_before.jpg'#+bbox_path.split('2020')[1]+'d.jpg'
after_save_path = 'pic_demo/' + video + '_after.jpg'#+bbox_path.split('2020')[1]+'.jpg'
cv2.imwrite(before_save_path, before)
cv2.imwrite(after_save_path, after)