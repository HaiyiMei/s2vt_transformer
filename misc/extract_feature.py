import os
import sys
import tqdm
import json
import glob
import argparse
import numpy as np
from PIL import Image
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from detectron2.layers.roi_align import ROIAlign 

def parse_args():

    parser = argparse.ArgumentParser(description='config of extraction')
    parser.add_argument('--dataset', dest='dataset',
                        default='MSRVTT',
                        help='Training dataset')
    parser.add_argument('--sample_len', dest='sample_len',
                        help='num of sample for a video', 
                        default=10, type=int)
    parser.add_argument('--clip_len', dest='clip_len',
                        help='num of frames of a clip', 
                        default=32, type=int)
    parser.add_argument('--crop_size', dest='crop_size',
                        help='crop size of frames', 
                        default=256, type=int)
    parser.add_argument('--model', dest='model',
                        help='model used to extract base feature',
                        default='resnet', type=str)
    parser.add_argument('--mode', dest='mode',
                        help='frame/clip mode',
                        default='frame', type=str)
    parser.add_argument('--box_num', dest='box_num', 
                        help='box_num for each frame',
                        default=10, type=int)
    parser.add_argument('--source_dir', dest='source_dir',
                        help='dir of the sources',
                        default='feats')
    parser.add_argument('--output_dir',
                        help='dir of the output',
                        default='uniform_clip', type=str)
    parser.add_argument('--threshold', dest='threshold',
                        help='threshold for filter the box score',
                        default=0.1, type=float)
    parser.add_argument('--nms', dest='nms',
                        help='nms for filter the box score',
                        default=0.5, type=float)
    args = parser.parse_args()
    return args

def build_incep_res():
    import pretrainedmodels
    model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
    model.cuda()
    model.eval()
    net = model.features

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        #if use TSM full_size, replace 224 with 256
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(),
        normalize,
    ])
    return net, transform

def build_resnet():
    resnet_ori = torchvision.models.resnet152(pretrained=True).cuda()
    resnet_base = nn.Sequential(*list(resnet_ori.children()))[:-2]
    # summary(resnet_base, (3, 224, 224))

    pretrained_dict = resnet_ori.state_dict()
    model_dict = resnet_base.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    resnet_base.load_state_dict(model_dict)

    resnet_base.cuda()
    resnet_base.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        #if use TSM full_size, replace 224 with 256
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(),
        normalize,
    ])
    return resnet_base, transform

def build_i3d():
    sys.path.append('/home/mhy/Documents/pytorch-resnet3d')
    from models.resnet import i3_res50_nl
    from util.util import clip_transform
    net_ori  = i3_res50_nl(400)
    state_dict = torch.load('/home/mhy/Documents/pytorch-resnet3d/pretrained/i3d_r50_nl_kinetics.pth')
    net_ori.load_state_dict(state_dict)
    net = nn.Sequential(*list(net_ori.children()))[:-3]

    pretrained_dict = net_ori.state_dict()
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    net.cuda()
    net.eval()

    transform = clip_transform('val', clip_len)
    return net, transform
    
def build_tsn():
    sys.path.append('/home/mhy/Documents/temporal-shift-module')
    from ops.models_modify import TSN
    # this_weights = '/home/mhy/Documents/temporal-shift-module/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth'
    this_weights = '/home/mhy/Documents/temporal-shift-module/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth'

    net = TSN(400, clip_len, 'RGB',
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Scale(256),
        #if use TSM full_size, replace 224 with 256
        transforms.CenterCrop(crop_size), 
        transforms.ToTensor(),
        normalize,
    ])
    return net, transform


def get_net():
    if model=='resnet':
        net, transform = build_resnet()
        tmp = torch.zeros(sample_len, 3, crop_size, crop_size).cuda()
    if model=='incep_res':
        net, transform = build_incep_res()
        tmp = torch.zeros(sample_len, 3, crop_size, crop_size).cuda()
    elif model=='i3d':
        net, transform = build_i3d()
        tmp = torch.zeros(sample_len, 3, clip_len, crop_size, crop_size).cuda()
    elif model=='tsn':
        net, transform = build_tsn()
        tmp = torch.zeros(sample_len, clip_len, 3, crop_size, crop_size).cuda()

    if mode=='frame':
        tmp = torch.zeros(sample_len, 3, crop_size, crop_size).cuda()
    return net, transform, tmp

def get_imgs():
    if model=='resnet' or model=='incep_res' or mode=='frame':
        imgs = [imgs_list[frame] for frame in frame_indices]
        imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs = [transform(img_) for img_ in imgs] # T*3*224*224
    else:
        clips = []
        for c in frame_indices:
            clip = imgs_list[c:c+clip_len]
            clip = [Image.open(img).convert('RGB') for img in clip]
            clips.append(clip)
        imgs = []
        for clip in clips:
            if model=='tsn':
                clip = [transform(clip_) for clip_ in clip] # T*3*224*224
                clip = torch.stack(clip, dim=0)
            elif model=='i3d':
                clip = transform(clip).permute(1, 0, 2, 3) # (3, T, 224, 224)
            imgs.append(clip)

    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.cuda()

    return imgs

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    sample_len = args.sample_len
    model = args.model
    box_num = args.box_num
    source_dir = args.source_dir
    threshold = args.threshold
    mode = args.mode
    crop_size = args.crop_size
    clip_len = args.clip_len
    print(args)

    frames_path = os.path.join(source_dir, dataset, 'frames')
    bbox_path = os.path.join(source_dir, dataset, 'bbox_conf_{}_nms_{}'.format(threshold, args.nms))
    features_path = os.path.join(source_dir, dataset, args.output_dir)
    config_file = features_path + '/config.json'

    feat_img_path = os.path.join(features_path, '{}_img').format(model)
    feat_box_path = os.path.join(features_path, '{}_box').format(model)

    print('frames path:', frames_path)
    print('bbox path:', bbox_path)
    print('saving features to:', features_path)

    os.makedirs(features_path, exist_ok=True)
    os.makedirs(feat_box_path, exist_ok=True)
    os.makedirs(feat_img_path, exist_ok=True)

    with open(config_file, 'w') as f:
        f.write(json.dumps(vars(args), indent=1))
    net, transform, tmp = get_net()
    output_size = (1, 1) if mode=='frame' else (1, 1, 1)
    avg_pool = torch.nn.AdaptiveAvgPool2d(output_size)
    max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))

    with torch.no_grad():
        print('input:', tmp.shape)
        tmp = net(tmp)
        feature_size = tmp.size(-1)
        print('output:', tmp.shape)
        print('feature map size: ' + str(feature_size))

    roi = ROIAlign((7, 7), feature_size/crop_size, 0, True)
    channel = 1536 if model=='incep_res' else 2048

    videos_list = os.listdir(frames_path)
    videos_list.sort()
    all_time = 0
    start_idx = 0
    for idx in tqdm.tqdm(range(start_idx, len(videos_list))):
        video = videos_list[idx]
        feat_file = video+'.pth'
        imgs_list = glob.glob(frames_path + '/' +video +'/*.jpg')
        imgs_list.sort()
        feat_len = len(imgs_list) - clip_len
        if feat_len<0:
            continue
        frame_indices = np.linspace(0, feat_len, sample_len, dtype=np.int)
        imgs = get_imgs()

        bbox_dets = torch.load(bbox_path+'/'+feat_file)  # frame_num*50*5
        bbox_dets = bbox_dets[frame_indices]
        
        frame_feat  = torch.zeros(sample_len, channel)
        box_feat = torch.zeros(box_num*sample_len, channel)
        with torch.no_grad():
            base_feat = net(imgs)
            # assert base_feat.size(-1) == crop_size // 16
            if model=='tsn' and mode=='clip':
                base_feat = base_feat.view((sample_len, clip_len) + base_feat.size()[1:])
                base_feat = base_feat.transpose(1, 2)  # (10, 2048, T, 7, 7)
            for i in range(sample_len):  
                bbox_det_single = bbox_dets[i]    # 50*5
                bbox_det_single = bbox_det_single[:box_num]
                bbox_det_single[:, 0] = 0

                if mode=='frame':
                    roi_feat = base_feat[i].unsqueeze(0).cpu()
                elif mode=='clip':
                    roi_feat = base_feat[i,:,0].unsqueeze(0).cpu()
                feat = roi(roi_feat, bbox_det_single.cpu())  # n, C, 16, 16
                feat = max_pool(feat).squeeze(-1).squeeze(-1)  # n, C
            
                box_feat[i*box_num:i*box_num+len(feat)] = feat
            base_feat = avg_pool(base_feat).squeeze(-1).squeeze(-1).squeeze(-1)
        frame_feat[:len(base_feat)] = base_feat 
        

        torch.save(frame_feat.cpu(), feat_img_path+'/'+video+'.pth')
        torch.save(box_feat.cpu(), feat_box_path+'/'+video+'.pth')
