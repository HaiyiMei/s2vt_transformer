import glob
import os
import argparse
import numpy as np
from PIL import Image
import sys
import tqdm
sys.path.append(os.getcwd())

import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from detectron2.layers.roi_align import ROIAlign 

import random
# from torchsummary import summary


def parse_args():

    parser = argparse.ArgumentParser(description='config of extraction')
    parser.add_argument('--dataset', dest='dataset',
                        default='MSRVTT',
                        help='Training dataset')
    parser.add_argument('--sample_len', dest='sample_len',
                        help='num of frames sample for a video', 
                        default=32, type=int)
    parser.add_argument('--model', dest='model',
                        help='model used to extract base feature',
                        default='resnet')
    parser.add_argument('--box_num', dest='box_num', 
                        help='box_num for each frame',
                        default=10, type=int)
    parser.add_argument('--source_dir', dest='source_dir',
                        help='dir of the sources',
                        default='/data/video_captioning/feats')
    parser.add_argument('--output_dir',
                        help='dir of the output',
                        default='uniform_frame', type=str)
    parser.add_argument('--threshold', dest='threshold',
                        help='threshold for filter the box score',
                        default=0.1, type=float)
    parser.add_argument('--nms', dest='nms',
                        help='nms for filter the box score',
                        default=0.5, type=float)
    
    args = parser.parse_args()

    return args

def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def build_resnet():
    resnet_ori = torchvision.models.resnet152(pretrained=True).cuda()
    resnet_base = nn.Sequential(*list(resnet_ori.children()))[:-2]
    # block = torchvision.models.resnet.BasicBlock
    # layer4 = resnet_base._make_layer(block, 512, 3, stride=1)
    # resnet_base.layer4 = layer4
    # summary(resnet_base, (3, 224, 224))

    pretrained_dict = resnet_ori.state_dict()
    model_dict = resnet_base.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    resnet_base.load_state_dict(model_dict)

    resnet_base.cuda()
    resnet_base.eval()
    return resnet_base

def build_tsn():
    sys.path.append('/home/mhy/Documents/temporal-shift-module')
    from ops.models_modify import TSN
    # this_weights = '/home/mhy/Documents/temporal-shift-module/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth'
    this_weights = '/home/mhy/Documents/temporal-shift-module/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth'
    net = TSN(400, sample_len, 'RGB',
                base_model='resnet50',
                consensus_type='avg',
                img_feature_dim='256',
                pretrain='imagenet',
                is_shift=True, shift_div=8, shift_place='blockres',
                non_local='_nl' in this_weights,
                )

    checkpoint = torch.load(this_weights)
    checkpoint = checkpoint['state_dict']

    net_dict = net.state_dict()
    checkpoint_dict = list(checkpoint)
    # print(checkpoint_dict)
    i = 0
    for key in list(net_dict.keys()):
        net_dict[key] = checkpoint[checkpoint_dict[i]]
        i += 1

    net.load_state_dict(net_dict)
    net.cuda()
    net.eval()
    return net
    

if __name__ == '__main__':
    # statistics of box_num
    max_box_per_frame = 0.0
    max_box_per_video = 0.0
    overflow = 0
    avg_box = 0.0

    args = parse_args()
    print(args)

    dataset = args.dataset
    sample_len = args.sample_len
    model = args.model
    box_num = args.box_num
    source_dir = args.source_dir
    threshold = args.threshold
    crop_size = 256
    area_threshold = 300 
    frames_path = os.path.join(source_dir, dataset, 'frames')
    bbox_path = os.path.join(source_dir, dataset, 'bbox_conf_{}_nms_{}'.format(threshold, args.nms))
    features_path = os.path.join(source_dir, dataset, args.output_dir)
    config_file = features_path + '/config.txt'

    feat_img_path = os.path.join(features_path, '{}_img').format(model)
    feat_box_path = os.path.join(features_path, '{}_box').format(model)
    # box_num_path = os.path.join(features_path, 'box_num')
    print(frames_path)
    print(bbox_path)
    print(features_path)
    mkdir(features_path)
    mkdir(feat_img_path)
    mkdir(feat_box_path)
    # mkdir(box_num_path)

    with open(config_file, 'w') as f:
        f.write('crop_size:{}\nsample_len: {}\nbox_num: {}\nthreshold:{}\narea_threshold: {}\nnms: {}\n'.format\
                                (str(crop_size), str(sample_len), str(box_num), str(threshold), str(area_threshold), str(args.nms)))

    if model=='resnet':
        net = build_resnet()
    elif model=='tsn':
        net = build_tsn()

    with torch.no_grad():
        tmp = net(torch.zeros(sample_len, 3, crop_size, crop_size).cuda())
        feature_size = tmp.size(-1)
        print('feature map size: ' + str(feature_size))

    roi = ROIAlign((7, 7), feature_size/crop_size, 0, True)
    max_pool2d = torch.nn.AdaptiveMaxPool2d((1, 1))
    avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
    # define preprocess
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform_resnet = transforms.Compose([
        transforms.Scale(256),
        #if use TSM full_size, replace 224 with 256
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(),
        normalize,
    ])

    videos_list = os.listdir(frames_path)
    videos_list.sort()
    all_time = 0
    start_idx = 0
    for idx in tqdm.tqdm(range(start_idx, len(videos_list))):
        video = videos_list[idx]
        # if video == 'ZN2_czSBSD0_240_250':
        #     continue
        feat_file = video+'.pth'
        now = time.time()
        imgs_list = glob.glob(frames_path + '/' +video +'/*.jpg')
        imgs_list.sort()
        video_len = len(imgs_list)
        frame_indices = np.linspace(0, video_len-1, sample_len, dtype=np.int)

        imgs = [imgs_list[frame] for frame in frame_indices]
        imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs_res = [transform_resnet(img_) for img_ in imgs] # T*3*224*224
        imgs_res = torch.stack(imgs_res, dim=0)
        imgs_res = imgs_res.cuda()

        bbox_dets = torch.load(bbox_path+'/'+feat_file)  # frame_num*50*5
        bbox_dets = bbox_dets[frame_indices]
        
        box_feat = torch.zeros(box_num*sample_len, 2048)
        # box_feat = []
        # box_num_rec = torch.zeros((sample_len))
        with torch.no_grad():
            base_feat = net(imgs_res)
            # assert base_feat.size(-1) == crop_size // 16
            for i in range(sample_len):  
                bbox_det_single = bbox_dets[i]    # 50*5
                # bbox_det_single = bbox_det_single[bbox_det_single[:, 0]>threshold]

                # # get rid of box area <=area_threshold
                # bbox_tmp = bbox_det_single
                # boxes = []
                # for j in range(len(bbox_tmp)):
                #     if (bbox_tmp[j, 3]-bbox_tmp[j,1]) * (bbox_tmp[j, 4]-bbox_tmp[j,2])>area_threshold:
                #         boxes.append(bbox_tmp[j])
                # if len(boxes)==0:
                #     continue
                # bbox_det_single = torch.stack(boxes)

                bbox_det_single = bbox_det_single[:box_num]
                bbox_det_single[:, 0] = 0

                feat = roi(base_feat[i].unsqueeze(0).cpu(), bbox_det_single.cpu())  # n, C, 16, 16
                feat = max_pool2d(feat).squeeze(-1).squeeze(-1)  # n, C
            
                box_feat[i*box_num:i*box_num+len(feat)] = feat
            base_feat = avg_pool2d(base_feat).squeeze(-1).squeeze(-1)
        

        torch.save(base_feat.cpu(), feat_img_path+'/'+video+'.pth')
        torch.save(box_feat.cpu(), feat_box_path+'/'+video+'.pth')