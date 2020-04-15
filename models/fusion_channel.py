import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, region_feat, frame_feat, mask=None):
        # region_feat: T, n, batch_size, d
        # frame_feat: T, batch_size, d
        # mask: T, n, batch_size
        att1 = self.encoder_attention(region_feat)
        att2 = self.decoder_attention(frame_feat)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(-1)  # T, n, batch_size
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        alpha = self.softmax(att)  # T, n, batch_size

        attention_weighted_encoding = (region_feat*alpha.unsqueeze(-1)).sum(1)
        return attention_weighted_encoding


class Fusion(nn.Module):

    def __init__(self, video_dim, box_dim, hidden_dim, mode='concat', n_layer=1):
        super(Fusion, self).__init__()
        self.video_dim = video_dim
        self.box_dim = box_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        self.dropout = nn.Dropout(0.1)

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.box_embed = nn.Linear(self.box_dim, self.hidden_dim)

        if 'channel' in self.mode:
            self.fc_reigion = nn.Linear(hidden_dim, hidden_dim)
            self.fc_reigion2 = nn.Linear(hidden_dim, hidden_dim)

            self.fc_frame = nn.Linear(hidden_dim, hidden_dim)
            self.fc_frame2 = nn.Linear(hidden_dim, hidden_dim)
        
        if 'att' in self.mode:
            self.attention = Attention(self.att_dim, self.att_dim, self.hidden_dim)

    def fc_mask(self, fc, input, mask=None):
        input = fc(self.dropout(input))  # N, batch_size, d
        if mask is not None:
            input.masked_fill_(mask.transpose(0, 1).unsqueeze(-1).expand(input.shape), 0)
        return input
    
    def channel(self, frame_feat, region_feat, mask=None, mask_frame=None):
        frame_feat = frame_feat.transpose(0, 1)
        batch_size, T, d = frame_feat.shape
        box_num = mask.size(1) // T
        frame_feat = frame_feat.unsqueeze(2).expand(-1 ,-1 ,box_num, -1).reshape(batch_size, T*box_num, d)
        frame_feat = frame_feat.masked_fill_(mask_frame.unsqueeze(-1).expand(-1, -1, d), 0)
        frame_feat = frame_feat.transpose(0, 1)

        if 'channelV1' in self.mode:
            ############ V1 ################
            region_weight = torch.mul(
                self.fc_mask(self.fc_reigion, region_feat, mask),
                self.fc_mask(self.fc_frame, frame_feat, mask_frame))
            region_weight = torch.sigmoid(region_weight)  # element-attention for every box

            frame_weight = torch.mul(
                self.fc_mask(self.fc_reigion2, region_feat, mask),
                self.fc_mask(self.fc_frame2, frame_feat, mask_frame))
            frame_weight = torch.sigmoid(frame_weight)  # element-attention for every box

            region_feat = torch.mul(region_feat, region_weight)
            frame_feat = torch.mul(frame_feat, frame_weight)

        elif 'channelV2' in self.mode:
            ############ V2 ################
            region_weight = torch.mul(
                self.fc_mask(self.fc_reigion, region_feat, mask),
                self.fc_mask(self.fc_frame, frame_feat, mask_frame))
            region_weight = torch.sigmoid(region_weight)  # element-attention for every box
            region_feat = torch.mul(region_feat, region_weight)
            # frame_weight = torch.mul(
            #     self.fc_mask(self.fc_reigion, region_feat),
            #     self.fc_mask(self.fc_frame, frame_feat))
            # frame_weight = torch.sigmoid(frame_weight)  # element-attention for every box
            # frame_feat = torch.mul(frame_feat, frame_weight)

        elif 'channelV3' in self.mode:
            ############ V3 ################
            region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat, mask))
            region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask))
            region_feat = torch.mul(region_feat, region_weight)

            frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat, mask_frame))
            frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight, mask_frame))
            frame_feat = torch.mul(frame_feat, frame_weight)

        return frame_feat, region_feat 

    def forward(self, frame_feat, region_feat):
        '''
        frame_feat: batch_size x T x d
        region_feat: batch_size x N x d
        mask: batch_size x N
        '''
        batch_size, T, _ = frame_feat.shape
        mask = region_feat.sum(-1).eq(0)  # batch_size x N
        mask_frame = mask.reshape(batch_size, T, -1)
        mask_frame[:, :, 0] = False
        mask_frame = mask_frame.reshape(batch_size, -1)

        frame_feat = frame_feat.transpose(0, 1)  # T x batch_size x d
        region_feat = region_feat.transpose(0, 1)  # N x batch_size x d

        frame_feat = self.fc_mask(self.img_embed, frame_feat, mask=None)
        region_feat = self.fc_mask(self.box_embed, region_feat, mask=mask)

        ###################

        if 'channel' in self.mode:
            frame_feat, region_feat = self.channel(frame_feat, region_feat, mask, mask_frame)
            feat = frame_feat + region_feat

            feat = feat.reshape(T, -1, batch_size, feat.size(-1))
            mask_frame = mask_frame.transpose(0, 1).reshape(T, -1, batch_size)
            feat = feat.sum(1) / (~mask_frame).sum(1).unsqueeze(-1).clamp(min=1)
        elif 'add' in self.mode:
            region_feat = region_feat.reshape(T, -1, batch_size, region_feat.size(-1))
            mask = mask.transpose(0, 1).reshape(T, -1, batch_size)
            region_feat = region_feat.sum(1) / (~mask).sum(1).unsqueeze(-1).clamp(min=1)
            feat = frame_feat + region_feat

        output = feat.transpose(0, 1)

        return output
