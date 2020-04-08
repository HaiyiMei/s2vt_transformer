import os
import numpy as np
import tqdm
import torch
import torch.nn.functional as F

def compute_iou(rec1, rec2):
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

features_in_path =  './feats/MSRVTT/uniform_batch/tsn_box/'
features_out_path =  './feats/MSRVTT/uniform_batch/tsn_box_noBatch/'
frames_path = './feats/MSRVTT/frames/'
bbox_path = './feats/MSRVTT/bbox_32_2020_nms_0.5'

T = 32
box_num = 10


if not os.path.isdir(features_out_path):
    os.mkdir(features_out_path)


feats_path = os.listdir(features_in_path)
feats_path.sort()

with torch.no_grad():
    for feat_file in tqdm.tqdm(feats_path):
        if '.pth' not in feat_file:
            continue
        if os.path.exists(features_out_path+'/'+feat_file):
            continue
        feat = torch.load(features_in_path+'/'+feat_file).cpu()  # N, d



        # feat_new = torch.zeros_like(feat)  # N, d

        # #######################3

        # imgs = os.listdir(frames_path + feat_file[:-4])
        # sample_frame = np.linspace(0, len(imgs)-1, 32, dtype=np.int)
        # cls_dets = torch.load(bbox_path+'/'+feat_file)

        # cls_det = cls_dets[sample_frame]
        # cls_det = cls_det[:, :box_num]  # 32, 10, 5

        # sim = F.normalize(feat, p=2, dim=-1)
        # sim = torch.matmul(sim, sim.transpose(0, 1))  # N, N
        # sim = sim[:box_num] # 10, N

        # box_n = cls_det[0] # 10, 5
        # iou = torch.zeros_like(sim) # 10, N

        # for i in range(1, T):
        #     for j in range(box_num):
        #         for k in range(box_num):
        #             iou[j, i*box_num+k] = compute_iou(box_n[j], cls_det[i, k])

        # sim += iou

        # ############################


        # # sim = F.normalize(feat, p=2, dim=-1)
        # # sim = torch.matmul(sim, sim.transpose(0, 1))  # N, N
        # # sim = sim[:box_num] # box, N

        # feat_new[:box_num] = feat[:box_num]

        # for i in range(1, T):
        #     feat_single = feat[i*box_num : (i+1)*box_num]  # box, d
        #     sim_single = sim[:, i*box_num : (i+1)*box_num]  # box, box

        #     for j in range(box_num):
        #         sim_ = sim_single[j]  # box
        #         _, idx = sim_.max(0)
        #         feat_new[i*box_num+j] = feat_single[idx]
        
        #################################

        # feat_new = feat.reshape(T, box_num, -1)
        # feat_new = feat_new.mean(0)

        #################################
        feat_new = feat[feat.sum(1)!=0]
        
        torch.save(feat_new, features_out_path+'/'+feat_file)



