import torch 
import torch.nn as nn
import torch.nn.functional as F

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
            attention_region = (region_feat*alpha.unsqueeze(-1)).sum(1)


        return attention_region

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        self.trans_encoder = opt["transformer_encoder"]
        self.hidden_dim = opt["dim_hidden"]
        self.video_dim = opt["dim_vid"]
        self.box_dim = opt["dim_box"]
        self.fusion = str(opt["fusion"])
        self.channel = opt["channel"]
        self.only_box = opt["only_box"]
        self.T = opt["sample_len"]

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.box_embed = nn.Linear(self.box_dim, self.hidden_dim)

        #### channel attention
        if self.channel:
            vector_dim = 1 if opt["scalar"] else self.hidden_dim

            self.fc_reigion = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.fc_reigion2 = nn.Linear(self.hidden_dim, vector_dim)

            self.fc_frame = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.fc_frame2 = nn.Linear(self.hidden_dim, vector_dim)

        if 'att' in self.fusion:
            self.attention = Attention(self.hidden_dim, self.hidden_dim)
        if self.trans_encoder:
            transformer_dim = self.hidden_dim*2 if 'concat' in self.fusion else self.hidden_dim
            encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4)
            self.trans_encoder = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(transformer_dim))

        self.dropout = nn.Dropout(0.1)

    def fc_mask(self, fc, input, mask=None):
        input = fc(self.dropout(input))  # batch_size, N, d
        # input = fc(input)  # batch_size, N, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input

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
        else:
            region_feat = region_feat.sum(1) / (~mask).sum(1).unsqueeze(-1).clamp(min=1)
        return region_feat

    def process_frame(self, frame_feat):
        '''
        frame_feat: batch_size x T x image_dim
        return frame_feat: T x batch_size, hidden_dim
        '''
        frame_feat = self.img_embed(self.dropout(frame_feat))
        if self.channel:
            frame_weight = torch.relu(self.fc_frame(frame_feat))
            frame_weight = torch.sigmoid(self.fc_frame2(frame_weight))
            frame_feat = torch.mul(frame_feat, frame_weight)
        mask = None

        # mask = frame_feat.sum(-1).eq(0)  # batch_size x T
        # if not mask.any():
        #     mask = None
        # frame_feat = self.fc_mask(self.img_embed, frame_feat, mask=mask)
        # if self.channel:
        #     frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat, mask))
        #     frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight, mask))
        #     frame_feat = torch.mul(frame_feat, frame_weight)

        # torch.save(frame_weight, 'generated_weight/frame/tmp.pth')
        frame_feat = frame_feat.transpose(0, 1)  # T, batch_size, d
        return frame_feat, mask

    def process_region(self, region_feat):
        '''
        reigon_feat: batch_size x N x image_dim
        return region_feat: N x batch_size, hidden_dim
        '''
        mask = region_feat.sum(-1).eq(0)  # batch_size x N
        region_feat = self.fc_mask(self.box_embed, region_feat, mask=mask)
        if self.channel:
            region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat, mask))
            region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask))
            region_feat = torch.mul(region_feat, region_weight)
            # torch.save(region_weight, 'generated_weight/region/tmp.pth')

        region_feat = region_feat.transpose(0, 1)  # N, batch_size, d
        return region_feat, mask

    def forward(self, frame_feat, region_feat=None):
        '''
        input_image: batch_size x T x image_dim
        input_box: batch_size x N x image_dim
        return input_feature: N x batch_size, hidden_dim
        '''
        if 'None' == self.fusion:
            if self.only_box:
                # frame_feat is region_feat in this situation
                region_feat, region_mask = self.process_region(frame_feat)
                encoder_out = self.align_region(region_feat, region_mask)
                frame_mask = encoder_out.sum(-1).eq(0)  # batch_size x N
            else:
                encoder_out, frame_mask  = self.process_frame(frame_feat)
        else:
            frame_feat, frame_mask = self.process_frame(frame_feat)
            region_feat, region_mask = self.process_region(region_feat)

            region_feat = self.align_region(region_feat, region_mask, frame_feat)
            if 'add' in self.fusion:
                encoder_out = frame_feat + region_feat
            if 'concat' in self.fusion:
                encoder_out = torch.cat([frame_feat, region_feat], dim=-1)

        # transformer encoder
        if self.trans_encoder:
            encoder_out = self.trans_encoder(encoder_out, src_key_padding_mask=frame_mask)

        return encoder_out, frame_mask

