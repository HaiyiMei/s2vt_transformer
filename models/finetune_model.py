import sys
import torch
import time
import os
import torch.nn as nn
import torchvision
from models.s2vt import S2VT


class finetune_model(nn.Module):
    def __init__(self, opt, evaluate=False):
        super(finetune_model, self).__init__()
        self.sample_len = opt["sample_len"]
        self.evaluate = evaluate

        if opt["model"] == 'tsn':
            self.img_encoder = self.build_tsn_dense()
        elif opt["model"] == 'resnet':
            self.img_encoder = self.build_resnet()
        self.transformer = S2VT(opt)
        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        if not self.eval:
            checkpoint = os.path.join(
                opt["save_path"], 'checkpoint', 'model_{}.pth'.format(opt["checkpoint_epoch"]))
            self.transformer.load_state_dict(state_dict=
                torch.load(checkpoint))

    def forward(self, input_image, input_caption=None, input_box=None, mode='train'):
        batch_size = input_image.size(0)
        input_image = input_image.reshape(batch_size*self.sample_len, 3, 256, 256)
        input_image = self.img_encoder(input_image)
        input_image = self.avg_pool2d(input_image).squeeze(-1).squeeze(-1)
        input_image = input_image.reshape(batch_size, self.sample_len, -1)

        seq_probs, seq_preds = self.transformer(
            input_image=input_image, 
            input_box=input_box,  # batch_size*(box_num_per_frame*frame_num)*2048
            input_caption=input_caption,
            mode=mode)
        
        return seq_probs, seq_preds

    def build_resnet(self):
        resnet_ori = torchvision.models.resnet101(pretrained=True).cuda()
        resnet_base = nn.Sequential(*list(resnet_ori.children()))[:-2]
        if not self.eval:
            pretrained_dict = resnet_ori.state_dict()
            model_dict = resnet_base.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            resnet_base.load_state_dict(model_dict)

        return resnet_base

    def build_tsn_dense(self):
        sys.path.append('/home/mhy/Documents/temporal-shift-module')
        from ops.models_modify import TSN
        this_weights = '/home/mhy/Documents/temporal-shift-module/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth'
        net = TSN(400, self.sample_len, 'RGB',
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
        return net