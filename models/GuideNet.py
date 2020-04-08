import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuideNet(nn.Module):

    def __init__(self, d, mode='weight', n_layer=1):
        super(GuideNet, self).__init__()
        self.d = d
        self.mode = mode
        if 'channel' in self.mode:
            d_h = d//2 if 'V3' in self.mode else d
            self.fc_reigion = nn.Linear(d, d_h)
            self.fc_reigion2 = nn.Linear(d_h, d)

            self.fc_frame = nn.Linear(d, d_h)
            self.fc_frame2 = nn.Linear(d_h, d)
            self.norm_region = nn.LayerNorm(d)
            self.norm_frame = nn.LayerNorm(d)
        
        elif 'weight' in self.mode:
            self.fc_reigion = nn.Linear(d, d)
            self.fc_frame = nn.Linear(d, d)
            self.full_att = nn.Linear(d, 1)
        elif 'att' in self.mode:
            self.multihead_attn = nn.MultiheadAttention(d, 4, dropout=0.1)
        
        if 'concat' in self.mode:
            self.fc_cat = nn.Linear(d*2, d)
        self.norm_fusion = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)

    def fc_mask(self, fc, input, mask=None):
        input = fc(self.dropout(input))  # batch_size, N, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input

    def forward(self, frame_feat, region_feat, mask=None):
        '''
        frame_feat: batch_size x T x d
        region_feat: batch_size x N x d
        '''
        if 'att' in self.mode:
            frame_feat_ = frame_feat.clone()
        batch_size, T, d = frame_feat.shape
        box_num = region_feat.size(1) // T
        frame_feat = frame_feat.unsqueeze(2).expand(-1 ,-1 ,box_num, -1).reshape(batch_size, T*box_num, d)
        frame_feat = frame_feat.masked_fill_(mask.unsqueeze(-1).expand(-1, -1, self.d), 0)

        if 'channelV1' in self.mode:
            ############ V1 ################
            region_weight = torch.mul(
                self.fc_mask(self.fc_reigion, region_feat, mask),
                self.fc_mask(self.fc_frame, frame_feat, mask))
            region_weight = torch.sigmoid(region_weight)  # element-attention for every box

            frame_weight = torch.mul(
                self.fc_mask(self.fc_reigion2, region_feat, mask),
                self.fc_mask(self.fc_frame2, frame_feat, mask))
            frame_weight = torch.sigmoid(frame_weight)  # element-attention for every box

            region_feat = torch.mul(region_feat, region_weight)
            frame_feat = torch.mul(frame_feat, frame_weight)

        elif 'channelV2' in self.mode:
            ############ V2 ################
            frame_weight = torch.mul(
                self.fc_mask(self.fc_reigion, region_feat, mask),
                self.fc_mask(self.fc_frame, frame_feat, mask))
            frame_weight = torch.sigmoid(frame_weight)  # element-attention for every box
            frame_feat = torch.mul(frame_feat, frame_weight)
            output = frame_feat

        elif 'channelV3' in self.mode:
            ############ V3 ################
            region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat, mask))
            region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask))
            region_feat = torch.mul(region_feat, region_weight)

            frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat, mask))
            frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight, mask))
            frame_feat = torch.mul(frame_feat, frame_weight)

        elif 'att' in self.mode:
            ############# V1 ##############
            # Q: frame (Txd)
            # K: region (Nxd)
            # V: frame (Txd)
            # out: region feature described by frame (Nxd)
            region_feat = self.multihead_attn(region_feat.transpose(0, 1), 
                                              frame_feat_.transpose(0, 1),
                                              frame_feat_.transpose(0, 1))[0]
            region_feat = region_feat.transpose(0, 1)

        elif 'weight' in self.mode:
            region_feat_ = self.fc_mask(self.fc_reigion, region_feat, mask)  # B*N*d
            frame_feat_ = self.fc_mask(self.fc_reigion, frame_feat, mask)  # B*N*d

            weight = torch.mul(region_feat_, frame_feat_)  # B*N*d
            weight = torch.sigmoid(self.full_att(weight))  #B*N*1
            output = weight * frame_feat + region_feat

            
        ########## concat or add #############
        if 'concat' in self.mode:
            output = torch.cat([frame_feat, region_feat], dim=-1)
            output = self.fc_mask(self.fc_cat, output, mask)
        elif 'add' in self.mode:
            output = frame_feat + region_feat

        # output = self.norm_fusion(output)



        return output
