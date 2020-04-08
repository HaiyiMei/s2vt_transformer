import os
import sys
import torch

sys.path.append(os.getcwd())

NMS = 0.3
filter_size = 500

features_in_path =  '/home/mhy/Documents/feats/MSRVTT/bbox/'
features_out_path =  '/home/mhy/Documents/feats/MSRVTT/bbox_filter/'

if not os.path.isdir(features_out_path):
    os.mkdir(features_out_path)

feats_path = os.listdir(features_in_path)
feats_path.sort()

with torch.no_grad():
    for idx, feat_file in enumerate(feats_path):
        if '.pth' not in feat_file:
            continue
        if os.path.exists(features_out_path+'/'+feat_file):
            print(idx+1, '/', len(feats_path), feat_file, 'pass')
            continue
        feat = torch.load(features_in_path+'/'+feat_file).cpu()
        feat_new = torch.zeros_like(feat)

        for i in range(feat.shape[0]):
            cls_dets = feat[i]
            tmp = []
            for cls_det in cls_dets:
                if (cls_det[3]-cls_det[1]) * (cls_det[4]-cls_det[2]) > filter_size:
                    tmp.append(cls_det)
            if len(tmp)==0:
                continue
            tmp = torch.stack(tmp)
            feat_new[i, :tmp.shape[0]] = tmp

        torch.save(feat_new, features_out_path+'/'+feat_file)

        print(idx+1, '/', len(feats_path), feat_file)

