import sys
sys.path.append("misc")
import utils
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from models.fusion_channel import Fusion

class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(Attention, self).__init__()

        self.region_fc = nn.Linear(hidden_dim, attention_dim)
        self.frame_fc = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, region_feat, frame_feat, mask=None):
        # region_feat: T, n, batch_size, d
        # frame_feat: T, batch_size, d
        # mask: T, n, batch_size

        att1 = self.region_fc(region_feat)
        att2 = self.frame_fc(frame_feat).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(-1)  # T, n, batch_size
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        alpha = self.softmax(att)  # T, n, batch_size

        attention_weighted_encoding = (region_feat*alpha.unsqueeze(-1)).sum(1)
        return attention_weighted_encoding

class Attention_two(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(Attention_two, self).__init__()

        self.region_fc = nn.Linear(hidden_dim, attention_dim)
        self.word_fc = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, region_feat, mask=None):
        # region_feat: T, n, batch_size, d
        # word_feat: batch_size, d
        # mask: T, n, batch_size

        att1 = self.region_fc(region_feat)
        att = self.full_att(self.relu(att1)).squeeze(-1)  # T, n, batch_size
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        alpha = self.softmax(att)  # T, n, batch_size

        attention_weighted_encoding = (region_feat*alpha.unsqueeze(-1)).sum(1)
        return attention_weighted_encoding

class Encoder_Transformer(nn.Module):
    def __init__(self, opt):
        super(Encoder_Transformer, self).__init__()
        
        self.hidden_dim = opt["dim_hidden"]
        self.video_dim = opt["dim_vid"]
        self.box_dim = opt["dim_box"]
        self.fusion = opt["fusion"]

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.box_embed = nn.Linear(self.box_dim, self.hidden_dim)

        #### channel attention
        self.fc_reigion = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_reigion2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc_frame = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_frame2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.encoder_dim = self.hidden_dim if 'add' in self.fusion else self.hidden_dim*2
        self.align = 'V1' in self.fusion or 'V2' in self.fusion
        if 'att' in self.fusion:
            if self.align:
                self.attention = Attention_two(self.encoder_dim, self.hidden_dim)
            else:
                self.attention = Attention(self.hidden_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.encoder_dim, nhead=4)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(self.encoder_dim))

        self.dropout = nn.Dropout(0.1)

    def channel(self, frame_feat, region_feat, mask=None, mask_frame=None):

        if 'channelV1' in self.fusion:
            ############ V1 ################
            frame_feat = frame_feat.transpose(0, 1)
            batch_size, T, d = frame_feat.shape
            box_num = mask.size(1) // T
            frame_feat = frame_feat.unsqueeze(2).expand(-1 ,-1 ,box_num, -1).reshape(batch_size, T*box_num, d)
            frame_feat = frame_feat.masked_fill_(mask_frame.unsqueeze(-1).expand(-1, -1, d), 0)
            frame_feat = frame_feat.transpose(0, 1)

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

        elif 'channelV2' in self.fusion:
            ############ V2 ################
            frame_feat = frame_feat.transpose(0, 1)
            batch_size, T, d = frame_feat.shape
            box_num = mask.size(1) // T
            frame_feat = frame_feat.unsqueeze(2).expand(-1 ,-1 ,box_num, -1).reshape(batch_size, T*box_num, d)
            frame_feat = frame_feat.masked_fill_(mask_frame.unsqueeze(-1).expand(-1, -1, d), 0)
            frame_feat = frame_feat.transpose(0, 1)

            region_weight = torch.mul(
                self.fc_mask(self.fc_reigion, region_feat, mask),
                self.fc_mask(self.fc_frame, frame_feat, mask_frame))
            region_weight = torch.sigmoid(region_weight)  # element-attention for every box
            region_feat = torch.mul(region_feat, region_weight)
    
        elif 'channelV3' in self.fusion:
            ############ V3 ################
            region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat, mask))
            region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask))
            region_feat = torch.mul(region_feat, region_weight)

            frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat))
            frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight))
            frame_feat = torch.mul(frame_feat, frame_weight)

        return frame_feat, region_feat 

    def fc_mask(self, fc, input, mask=None):
        input = fc(self.dropout(input))  # N, batch_size, d
        if mask is not None:
            input.masked_fill_(mask.transpose(0, 1).unsqueeze(-1).expand(input.shape), 0)
        return input

    def align_region(self, region_feat, frame_feat=None, mask=None):
        if 'att' in self.fusion:
            if frame_feat is not None:
                region_feat = self.attention(region_feat, frame_feat, mask)
            else:
                region_feat = self.attention(region_feat, mask)
        else:
            region_feat = region_feat.sum(1) / (~mask).sum(1).unsqueeze(-1).clamp(min=1)
        return region_feat

    def forward(self, frame_feat, region_feat):
        '''
        input_image: batch_size x T x image_dim
        input_box: batch_size x N x box_dim
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

        if 'channel' in self.fusion:
            frame_feat, region_feat = self.channel(frame_feat, region_feat, mask, mask_frame)

        if self.align:
            if 'add' in self.fusion:
                input_feature = frame_feat + region_feat
            if 'concat' in self.fusion:
                input_feature = torch.cat([frame_feat, region_feat], dim=-1)
            input_feature = input_feature.reshape(T, -1, batch_size, input_feature.size(-1))
            mask_frame = mask_frame.transpose(0, 1).reshape(T, -1, batch_size)
            input_feature = self.align_region(input_feature, mask=mask_frame)
        else:
            mask = mask.transpose(0, 1).reshape(T, -1, batch_size)
            region_feat = region_feat.reshape(T, -1, batch_size, region_feat.size(-1))
            region_feat = self.align_region(region_feat, frame_feat, mask)
            if 'add' in self.fusion:
                input_feature = frame_feat + region_feat
            if 'concat' in self.fusion:
                input_feature = torch.cat([frame_feat, region_feat], dim=-1)

        # transformer encoder
        input_feature = self.trans_encoder(input_feature)

        return input_feature, None
