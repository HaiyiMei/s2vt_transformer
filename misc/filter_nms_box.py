import os
import sys
import torch
from model.roi_layers import nms

sys.path.append(os.getcwd())

NMS = 0.3

features_in_path =  '/home/mei/Documents/feats/MSRVTT/bbox/'
features_out_path =  '/home/mei/Documents/feats/MSRVTT/bbox_filter/'

if not os.path.isdir(features_out_path):
    os.mkdir(features_out_path)

feats_path = os.listdir(features_in_path)

with torch.no_grad():
    for idx, feat_file in enumerate(feats_path):
        if '.pth' not in feat_file:
            continue
        feat = torch.load(features_in_path+'/'+feat_file).cpu()
        feat_new = torch.zeros_like(feat)

        for i in range(feat.shape[0]):
            cls_det = feat[i]
            keep = nms(cls_det.cpu(), cls_det[:,-1].cpu(), NMS)
            cls_det = cls_det[keep]

            feat_new[i, :cls_det.shape[0]] = cls_det

        torch.save(feat_new, features_out_path+'/'+feat_file)

        print(idx+1, '/', len(feats_path), feat_file)

