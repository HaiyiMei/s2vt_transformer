import torch 
import torch.nn as nn
import torch.nn.functional as F


def fc_mask(fc, input, mask=None):
    input = fc(F.dropout(input, p=0.1))  # batch_size, N, d
    # input = fc(input)  # batch_size, N, d
    if mask is not None:
        input.masked_fill_(mask.unsqueeze(-1), 0.)
        # input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
    return input


class WCM(nn.Module):
    def __init__(self, hidden_dim, vector_dim, wcm_len):
        super(WCM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.wcm_len = wcm_len
        
        self.fc1 = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(wcm_len)])
        self.fc2 = nn.ModuleList([nn.Linear(self.hidden_dim, self.vector_dim) for _ in range(wcm_len)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, feat, mask):
        for l1, l2 in zip(self.fc1, self.fc2):
            weight = self.relu(fc_mask(l1, feat, mask))
            weight = self.sigmoid(fc_mask(l2, weight, mask))
            feat = torch.mul(feat, weight)
        return feat


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(Attention, self).__init__()

        # self.region_fc = nn.Linear(hidden_dim, attention_dim)
        # self.frame_fc = nn.Linear(hidden_dim, attention_dim)
        self.att = nn.Linear(2*hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, region_feat, frame_feat, mask=None, att_max=False):
        # region_feat: T, n, batch_size, d
        # frame_feat: T, batch_size, d
        # mask: T, n, batch_size

        ######################
        # att1 = self.region_fc(region_feat)
        # att2 = self.frame_fc(frame_feat).unsqueeze(1)
        # att = self.full_att(self.relu(att1 + att2)).squeeze(-1)  # T, n, batch_size
        ######################
        frame_feat = frame_feat.unsqueeze(1).repeat(1, region_feat.size(1), 1, 1)
        att = self.full_att(torch.tanh(self.att(
            torch.cat([region_feat, frame_feat], dim=-1)
        ))).squeeze(-1)
        ######################
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        alpha = self.softmax(att)  # T, n, batch_size

        if att_max:
            T, n, B, d = region_feat.shape
            _, max_idx = alpha.max(1)  # T, batch_size
            region_feat = region_feat.transpose(1, 2).reshape(T*B, n, d)
            attention_region = region_feat[torch.arange(T*B), max_idx.view(-1)]
            attention_region = attention_region.reshape(T, B, d)
        else:
            attention_region = (region_feat*alpha.unsqueeze(-1)).sum(1) / (~mask).sum(1).unsqueeze(-1).clamp(min=1)

        return attention_region

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        self.opt = opt
        self.trans_encoder = opt["transformer_encoder"]
        self.hidden_dim = opt["dim_hidden"]
        self.video_dim = opt["dim_vid"]
        self.box_dim = opt["dim_box"]
        self.fusion = str(opt["fusion"])
        self.wcm = str(opt["wcm"])
        self.only_box = opt["only_box"]
        self.T = opt["sample_len"]

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.box_embed = nn.Linear(self.box_dim, self.hidden_dim)

        #### channel attention
        if self.wcm == 'channel' or self.wcm == 'scalar':
            vector_dim = 1 if self.wcm == "scalar" else self.hidden_dim
            self.wcm_region = WCM(self.hidden_dim, vector_dim, opt["wcm_len"])
            self.wcm_frame = WCM(self.hidden_dim, vector_dim, opt["wcm_len"])

        if self.wcm == 'fc':
            self.fc_region = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.fc_frame = nn.Linear(self.hidden_dim, self.hidden_dim)

            self.wcm_region = lambda feat, mask: fc_mask(self.fc_region, feat, mask)
            self.wcm_frame = lambda feat, mask: fc_mask(self.fc_frame, feat, mask)

        transformer_dim = self.hidden_dim

        if 'att' in self.fusion:
            self.attention = Attention(self.hidden_dim, self.hidden_dim)
        if 'cbp' in self.fusion:
            from compact_bilinear_pooling import CompactBilinearPooling
            self.cbp = CompactBilinearPooling(self.hidden_dim, self.hidden_dim, opt["cbp_dim"]).cuda()
            transformer_dim = 4000
        if 'concat' in self.fusion:
            transformer_dim = self.hidden_dim * 2

        if self.trans_encoder:
            encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4)
            self.trans_encoder = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(transformer_dim))

    def align_region(self, region_feat, mask, frame_feat=None):
        '''
        reigon_feat: N x batch_size x image_dim
        mask: batch_size x N
        frame_feat: T x batch_size x image_dim
        return region_feat: T x batch_size x hidden_dim
        '''
        batch_size = mask.size(0)
        mask = mask.transpose(0, 1).reshape(self.T, -1, batch_size)
        region_feat = region_feat.reshape(self.T, -1, batch_size, self.hidden_dim)
        if 'att' in self.fusion and frame_feat is not None:
            region_feat = self.attention(
                region_feat, frame_feat, mask=mask, att_max='max' in self.fusion)
        elif 'max' in self.fusion:
            region_feat.masked_fill_(mask.unsqueeze(-1), -1e9)
            region_feat = region_feat.max(1)[0]
        else:
            region_feat = region_feat.sum(1) / (~mask).sum(1).unsqueeze(-1).clamp(min=1)
        return region_feat

    def process_feat(self, feat, is_region=False):
        if is_region:
            mask = feat.sum(-1).eq(0)  # batch_size x N
            feat = fc_mask(self.box_embed, feat, mask=mask)
            if self.opt["wcm"] != 'None':
                feat = self.wcm_region(feat, mask)
        else:
            mask = None
            feat = fc_mask(self.img_embed, feat, mask=mask)
            if self.opt["wcm"] != 'None':
                feat = self.wcm_frame(feat, mask)

        feat = feat.transpose(0, 1)  # T, batch_size, d
        return feat, mask

    def forward(self, frame_feat, region_feat=None):
        '''
        frame_feat: batch_size x T x image_dim
        region_feat: batch_size x N x image_dim
        return input_feature: N x batch_size, hidden_dim
        '''
        if 'None' == self.fusion:
            if self.only_box:
                # frame_feat is region_feat in this situation
                region_feat, region_mask = self.process_feat(frame_feat)
                encoder_out = self.align_region(region_feat, region_mask)
                frame_mask = encoder_out.sum(-1).eq(0)  # batch_size x N
            else:
                encoder_out, frame_mask  = self.process_feat(frame_feat)
        else:
            frame_feat, frame_mask = self.process_feat(frame_feat)
            region_feat, region_mask = self.process_feat(region_feat, is_region=True)

            region_feat = self.align_region(region_feat, region_mask, frame_feat)
            if 'add' in self.fusion:
                encoder_out = frame_feat + region_feat
            if 'concat' in self.fusion:
                encoder_out = torch.cat([frame_feat, region_feat], dim=-1)
            if 'mul' in self.fusion:
                encoder_out = frame_feat * region_feat
            if 'cbp' in self.fusion:
                encoder_out = self.cbp(frame_feat, region_feat)

        # transformer encoder
        if self.trans_encoder:
            encoder_out = self.trans_encoder(encoder_out, src_key_padding_mask=frame_mask)

        return encoder_out, frame_mask

