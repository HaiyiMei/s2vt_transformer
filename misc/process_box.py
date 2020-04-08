import torch
import numpy as np
import os


def cul_simularity(X):
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    sim = torch.matmul(X, X.t())
    return sim

bbox_path = '../box_after_filter'
# GCN_path = './GCN_X'
GCN_withN_path = '../resnet_GCN_X_N'
GCN_gather = '../resnet_GCN_X_gather'
resnet_path = '../resnet_GCN_X'
GCN_path = '../resnet_GCN_X'

# videos = os.listdir(bbox_path)
# for idx in range(len(videos)):
#     video = videos[idx]
#     print(idx+1, '/', len(videos))
#     if not os.path.exists(GCN_path+'/'+video):
#         continue
#     cls_dets = torch.load(bbox_path+'/'+video)
#     GCN_X = torch.load(GCN_path+'/'+video)

#     list_N = [0]
#     sample_frame = np.arange(0, cls_dets.shape[0]-32, 16, dtype=np.int)
#     sample_frame = sample_frame[:10]
#     cls_det = cls_dets[sample_frame]
#     for num in range(cls_det.shape[0]):
#         tmp_cls = cls_det[num]
#         tmp_cls = tmp_cls[tmp_cls.sum(1)>0]
#         no_zero = tmp_cls.shape[0]
#         list_N.append(list_N[-1]+no_zero)
    
#     dict_N = {'GCN_X':GCN_X, 'N': list_N}
#     torch.save(dict_N, GCN_withN_path+'/'+video)

videos = os.listdir(GCN_withN_path)
for idx in range(len(videos)):
    video = videos[idx]
    print(idx+1, '/', len(videos))
    input_gcn = torch.load(GCN_withN_path+'/'+video)
    resnet_X = torch.load(resnet_path+'/'+video)
    gcn = input_gcn['GCN_X']
    N = input_gcn['N']
    N = list(set(N))
    N = sorted(N)

    if len(N)==2:
        torch.save(gcn, GCN_gather+'/'+video)
        continue

    sim = cul_simularity(resnet_X)
    nearest_neighbor = torch.LongTensor(len(gcn)).fill_(-1)

    for idx in range(len(N)-2):
        now = N[idx]  # frame nums for n-1
        after = N[idx+1]  # frame nums for n
        for num in range(now, after):
            sim_idx = sim[num]
            sim_idx[:after] = 0  # this frame and before it is not selectable
            sort, order = torch.sort(sim_idx, descending=True)
            nearest_neighbor[num] = order[0].item()
            idx_range = len(sort) if len(sort)<10 else 10
            for i in range(idx_range):
                if sort[i]>0.9 and N[idx+1]<=order[i]<=N[idx+2]:
                    nearest_neighbor[num] = order[i].item()
                    break
    # print(nearest_neighbor)
    new_gcn = []
    mask = torch.zeros(len(gcn))
    for box_idx in range(len(gcn)):
        if mask[box_idx]==1:
            continue
        box = box_idx
        boxes = []
        while box != -1:
            boxes.append(box)
            box = nearest_neighbor[box].item()

        mask[boxes] = 1
        new_gcn.append(gcn[boxes].mean(0))
        order = boxes
    new_gcn = torch.stack(new_gcn)

    torch.save(new_gcn, GCN_gather+'/'+video)