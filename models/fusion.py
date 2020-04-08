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

    def __init__(self, d, h_d, mode='concat', n_layer=1):
        super(Fusion, self).__init__()
        self.d = d
        self.h_d = h_d
        self.mode = mode

        self.dropout = nn.Dropout(0.1)

        if 'concat' in self.mode:
            dim = d
            self.fc_cat = nn.Linear(dim*2, h_d)
        if 'encoder' in self.mode:
            self.img_embed = nn.Linear(self.d, self.h_d)
            self.box_embed = nn.Linear(self.d, self.h_d)
            dim = h_d

            # self.fc_cat = nn.Linear(h_d*2, h_d)
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4)
            self.encoder_box = nn.TransformerEncoder(encoder_layer, n_layer)
            self.encoder_img = nn.TransformerEncoder(encoder_layer, n_layer)

        if 'decoder' in self.mode:
            self.img_embed = nn.Linear(self.d, self.h_d)
            self.box_embed = nn.Linear(self.d, self.h_d)

            dim = h_d

            decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=4)
            self.decoder = nn.TransformerDecoder(decoder_layer, n_layer)
        
        if 'att' in self.mode:
            self.attention = Attention(dim, dim, self.h_d)

    def fc_mask(self, fc, input, mask=None):
        input = fc(self.dropout(input))  # N, batch_size, d
        if mask is not None:
            input.masked_fill_(mask.transpose(0, 1).unsqueeze(-1).expand(input.shape), 0)
        return input
    

    def forward(self, frame_feat, region_feat, mask=None):
        '''
        frame_feat: batch_size x T x d
        region_feat: batch_size x N x d
        mask: batch_size x N
        '''
        frame_feat = frame_feat.transpose(0, 1)  # T x batch_size x d
        region_feat = region_feat.transpose(0, 1)  # N x batch_size x d

        ###################

        if 'encoder' in self.mode:
            frame_feat = self.fc_mask(self.img_embed, frame_feat, mask=None)
            region_feat = self.fc_mask(self.box_embed, region_feat, mask=mask)
            if 'img' in self.mode:
                frame_feat = self.encoder_img(frame_feat)
            if 'box' in self.mode:
                region_feat = self.encoder_box(region_feat, src_key_padding_mask=mask)
                region_feat = region_feat.masked_fill_(mask.transpose(0, 1).unsqueeze(-1).expand(-1, -1, self.h_d), 0.)
        if 'decoder' in self.mode:
            frame_feat = self.fc_mask(self.img_embed, frame_feat, mask=None)
            region_feat = self.fc_mask(self.box_embed, region_feat, mask=mask)
            output = self.decoder(frame_feat, 
                                  region_feat,
                                  memory_key_padding_mask=mask)
            output = output.transpose(0, 1)
            return output

        T, batch_size, d = frame_feat.shape


        region_feat = region_feat.reshape(T, -1, batch_size, d)
        mask = mask.transpose(0, 1).reshape(T, -1, batch_size)

        if 'att' in self.mode:
            region_feat = self.attention(region_feat, frame_feat, mask)
        else:
            region_feat = region_feat.sum(1) / (~mask).sum(1).unsqueeze(-1).clamp(min=1)

        output = torch.cat([frame_feat, region_feat], dim=-1)
        if 'concat' in self.mode:
            output = self.fc_cat(self.dropout(output))

        output = output.transpose(0, 1)

        return output
