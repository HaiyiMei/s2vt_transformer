import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):

    def __init__(self, hidden_dim, mode='attV1', n_layer=1):
        super(Fusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode

        if 'channel' in self.mode:
            d_h = hidden_dim//2 if 'V3' in self.mode else hidden_dim
            self.fc_reigion = nn.Linear(hidden_dim, d_h)
            self.fc_reigion2 = nn.Linear(d_h, hidden_dim)

            self.fc_frame = nn.Linear(hidden_dim, d_h)
            self.fc_frame2 = nn.Linear(d_h, hidden_dim)
        
        if 'att' in self.mode:
            self.multihead_attn = nn.MultiheadAttention(self.hidden_dim, 4, dropout=0.1)
        if 'decoder' in self.mode:
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4)
            self.decoder = nn.TransformerDecoder(decoder_layer, n_layer)
        if 'encoder' in self.mode:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=4)
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)
            self.encoder1 = nn.TransformerEncoder(encoder_layer, n_layer)
        
        # self.norm_fusion = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)

        if 'attV2' == self.mode or 'attV3' == self.mode:
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)

            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        if 'attV5' == self.mode:
            self.multihead_attn2 = nn.MultiheadAttention(self.hidden_dim, 4, dropout=0.1)
        if 'concat' in self.mode:
            self.fc_cat = nn.Linear(hidden_dim*2, hidden_dim)

    def fc_mask(self, fc, input, mask=None):
        input = fc(self.dropout(input))  # batch_size, N, d
        if mask is not None:
            input.masked_fill_(mask.transpose(0, 1).unsqueeze(-1).expand(input.shape), 0)
        return input
    
    def channel(self, frame_feat, region_feat, mask):
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
            torch.save(frame_weight, 'frame_weight.pth')

        elif 'channelV3' in self.mode:
            ############ V3 ################
            region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat, mask))
            region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask))
            region_feat = torch.mul(region_feat, region_weight)

            frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat, mask))
            frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight, mask))
            frame_feat = torch.mul(frame_feat, frame_weight)

        return frame_feat, region_feat

    def channel_single(self, frame_feat, region_feat, mask):
        frame_feat_ori = self.frame_feat_ori
        T = self.T
        box_num = self.box_num

        if 'channelV1' in self.mode:
            ############ V1 ################
            frame_feat_att = []
            region_feat_att = []
            for i in range(T):
                frame_feat_single = frame_feat[i*box_num : (i+1) * box_num]
                region_feat_single = region_feat[i*box_num : (i+1) * box_num]
                mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

                region_weight = torch.mul(
                    self.fc_mask(self.fc_reigion, region_feat_single, mask_single),
                    self.fc_mask(self.fc_frame, frame_feat_single, mask_single))
                region_weight = torch.sigmoid(region_weight)  # element-attention for every box

                frame_weight = torch.mul(
                    self.fc_mask(self.fc_reigion2, region_feat_single, mask_single),
                    self.fc_mask(self.fc_frame2, frame_feat_single, mask_single))
                frame_weight = torch.sigmoid(frame_weight)  # element-attention for every box

                region_feat_single = torch.mul(region_feat_single, region_weight)
                frame_feat_single = torch.mul(frame_feat_single, frame_weight)

                frame_feat_att.append(frame_feat_single)
                region_feat_att.append(region_feat_single)

            frame_feat = torch.cat(frame_feat_att, dim=0)  # N, batch_size, d
            region_feat = torch.cat(region_feat_att, dim=0)  # N, batch_size, d

        elif 'channelV3' in self.mode:
            ############ V3 ################
            frame_feat_att = []
            region_feat_att = []
            for i in range(T):
                frame_feat_single = frame_feat[i*box_num : (i+1) * box_num]
                region_feat_single = region_feat[i*box_num : (i+1) * box_num]
                mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

                region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat_single, mask_single))
                region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask_single))
                region_feat_single = torch.mul(region_feat_single, region_weight)

                frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat_single, mask_single))
                frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight, mask_single))
                frame_feat_single = torch.mul(frame_feat_single, frame_weight)

                frame_feat_att.append(frame_feat_single)
                region_feat_att.append(region_feat_single)

                if i == 1:
                    region_weight_tmp = region_weight.clone()
                    region_weight_tmp.masked_fill_(mask_single.transpose(0, 1).unsqueeze(-1).expand(region_weight_tmp.shape), 0)
                    torch.save(frame_weight, 'frame_weight_'+self.mode+'.pth')
                    torch.save(region_weight_tmp, 'region_weight_'+self.mode+'.pth')

            frame_feat = torch.cat(frame_feat_att, dim=0)  # N, batch_size, d
            region_feat = torch.cat(region_feat_att, dim=0)  # N, batch_size, d
            torch.save(frame_weight, 'frame_weight_'+self.mode+'.pth')
            torch.save(region_weight, 'region_weight'+self.mode+'.pth')
            

        elif 'channelV4' in self.mode:
            ############ V3 ################
            frame_feat_att = []
            region_feat_att = []
            for i in range(T):
                frame_feat_single = frame_feat_ori[i].unsqueeze(0)
                region_feat_single = region_feat[i*box_num : (i+1) * box_num]
                mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

                region_weight = torch.relu(self.fc_mask(self.fc_reigion, region_feat_single, mask_single))
                region_weight = torch.sigmoid(self.fc_mask(self.fc_reigion2, region_weight, mask_single))
                tmp = torch.mul(region_feat_single, region_weight)
                tmp = tmp.masked_fill_(mask_single.transpose(0, 1).unsqueeze(-1).expand(-1, -1, self.hidden_dim), 0)
                tmp = tmp.sum(0) / (~mask_single).sum(-1).unsqueeze(-1).clamp(min=1)

                region_feat_att.append(tmp.unsqueeze(0))

                frame_weight = torch.relu(self.fc_mask(self.fc_frame, frame_feat_single))
                frame_weight = torch.sigmoid(self.fc_mask(self.fc_frame2, frame_weight))
                frame_feat_single = torch.mul(frame_feat_single, frame_weight)

                frame_feat_att.append(frame_feat_single)

                if i == 1:
                    region_weight_tmp = region_weight.clone()
                    region_weight_tmp.masked_fill_(mask_single.transpose(0, 1).unsqueeze(-1).expand(region_weight_tmp.shape), 0)
                    torch.save(frame_weight, 'frame_weight_'+self.mode+'.pth')
                    torch.save(region_weight_tmp, 'region_weight_'+self.mode+'.pth')

            frame_feat = torch.cat(frame_feat_att, dim=0)  # N, batch_size, d
            region_feat = torch.cat(region_feat_att, dim=0)  # N, batch_size, d
            mask = None

        return frame_feat, region_feat, mask

    def attention(self, frame_feat, region_feat, mask):
        frame_feat_ori = self.frame_feat_ori
        T = self.T
        box_num = self.box_num

        if 'decoderV1' in self.mode:
            ############# V1 ##############
            # Q: frame (Txd)
            # K: region (Nxd)
            # V: frame (Txd)
            # out: region feature described by frame (Nxd)
            output = self.decoder(region_feat,
                                    frame_feat_ori)
        
        elif 'decoderV2' in self.mode:
            ############# V2 ##############
            # Q: region (Nxd)
            # K: frame (Txd)
            # V: region (Nxd)
            # out: frame feature described by region (Txd)
            output = self.decoder(frame_feat_ori, 
                                    region_feat,
                                    memory_key_padding_mask=mask)

        elif 'decoderV3' in self.mode:
            ############# V3 ##############
            # Q: region (nxd)
            # K: frame (1xd)
            # V: region (nxd)
            # out: frame feature described by region (1xd) -> (Txd)
            region_feat_att = []
            for i in range(T):
                frame_feat_single = frame_feat_ori[i].unsqueeze(0)
                region_feat_single = region_feat[i*box_num : (i+1) * box_num]
                mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

                tmp = self.decoder(frame_feat_single, 
                                region_feat_single,
                                memory_key_padding_mask=mask_single)
                region_feat_att.append(tmp)
            output = torch.cat(region_feat_att, dim=0)  # T, batch_size, d
    
        elif 'encoderV4' in self.mode:
            ############# V4 ##############
            # Q: region (nxd)
            # K: region (nxd)
            # V: region (nxd)
            # out: frame feature described by region (nxd) -> (Nxd)
            # out: encode region features of each frame, then concat and sum with frame features -> (N×d)
            region_feat_att = []
            for i in range(T):
                region_feat_single = region_feat[i*box_num : (i+1) * box_num]
                mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

                tmp = self.encoder(region_feat_single, src_key_padding_mask=mask_single)
                region_feat_att.append(tmp)
            
            region_feat_att = torch.cat(region_feat_att, dim=0)  # N, batch_size, d
            region_feat_att_1 = self.encoder1(region_feat, src_key_padding_mask=mask)
            region_feat = region_feat_att + region_feat_att_1
            return frame_feat, region_feat
            # output = frame_feat + region_feat
    
        elif 'encoderV5' in self.mode:
            ############# V5 ##############
            # Q: region (nxd)
            # K: region (nxd)
            # V: region (nxd)
            # out: frame feature described by region (nxd), then Avgpooling (1,d) -> (Txd)
            # out: encode region features of each frame, then Avgpooling and concat -> (T×d)
            region_feat_att = []
            for i in range(T):
                region_feat_single = region_feat[i*box_num : (i+1) * box_num]
                mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

                tmp = self.encoder(region_feat_single, src_key_padding_mask=mask_single)
                tmp = tmp.masked_fill_(mask_single.transpose(0, 1).unsqueeze(-1).expand(-1, -1, self.hidden_dim), 0)
                tmp = tmp.sum(0) / (~mask_single).sum(-1).unsqueeze(-1).clamp(min=1)
                region_feat_att.append(tmp.unsqueeze(0))
            region_feat_att = torch.cat(region_feat_att, dim=0)  # T, batch_size, d
            return frame_feat_ori, region_feat_att
            # output = frame_feat_ori + region_feat_att
        return output

    def forward(self, frame_feat, region_feat, mask=None):
        '''
        frame_feat: batch_size x T x d
        region_feat: batch_size x N x d
        mask: batch_size x N
        '''
        frame_feat_ori = frame_feat.clone()
        batch_size, T, d = frame_feat.shape
        box_num = region_feat.size(1) // T
        frame_feat = frame_feat.unsqueeze(2).expand(-1 ,-1 ,box_num, -1).reshape(batch_size, T*box_num, d)
        frame_feat = frame_feat.masked_fill_(mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim), 0)

        frame_feat = frame_feat.transpose(0, 1)  # N x batch_size x d
        frame_feat_ori = frame_feat_ori.transpose(0, 1)  # T x batch_size x d
        region_feat = region_feat.transpose(0, 1)  # N x batch_size x d

        self.T, self.box_num, self.frame_feat_ori = T, box_num, frame_feat_ori

        if 'channel' in self.mode and 'single' not in self.mode:
            frame_feat, region_feat = self.channel(frame_feat, region_feat, mask)

        if 'encoder' in self.mode or 'decoder' in self.mode:
            if 'encoderV4' in self.mode:
                frame_feat, region_feat = self.attention(frame_feat, region_feat, mask)
            else:
                output = self.attention(frame_feat, region_feat, mask)
    
        if 'single_channel' in self.mode:
            frame_feat, region_feat, mask = self.channel_single(frame_feat, region_feat, mask)
        
        if 'concat' in self.mode:
            output = torch.cat([frame_feat, region_feat], dim=-1)
            # output = self.fc_mask(self.fc_cat, output, mask)
        elif 'add' in self.mode:
            output = frame_feat + region_feat
        elif 'of' in self.mode:
            output = frame_feat


        output = output.transpose(0, 1)


        return output

        # elif 'attV1' in self.mode:
        #     ############# V1 ##############
        #     # Q: region (N×d)
        #     # K: frame (Txd)
        #     # V: frame (Txd)
        #     # out: region feature described by frame (Nxd)
        #     frame_feat_att = self.multihead_attn(region_feat,
        #                                           frame_feat_ori,
        #                                           frame_feat_ori)[0]
        #     output = region_feat + frame_feat_att

        # elif 'attV2' in self.mode:
        #     ############# V2 ##############
        #     # Q: frame (Txd)
        #     # K: region (Nxd)
        #     # V: region (Nxd)
        #     # out: frame feature described by all the regions (Txd)
        #     region_feat_att = self.multihead_attn(frame_feat_ori, 
        #                                           region_feat,
        #                                           region_feat,
        #                                           key_padding_mask=mask)[0]
        #     output = frame_feat_ori + self.dropout(region_feat_att)
        #     output = self.norm1(output)
        #     output2 = self.linear2(self.dropout(F.relu(self.linear1(output))))
        #     output = output + self.dropout(output2)
        #     output = self.norm2(output)

        # elif 'attV3' in self.mode:
        #     ############# V3 ##############
        #     # Q: region (nxd)
        #     # K: frame (1xd)
        #     # V: region (nxd)
        #     # out: each frame feature described by its corresponding region (1xd) -> (Txd)
        #     region_feat_att = []
        #     for i in range(T):
        #         frame_feat_single = frame_feat_ori[i].unsqueeze(0)
        #         region_feat_single = region_feat[i*box_num : (i+1) * box_num]
        #         mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

        #         tmp = self.multihead_attn(frame_feat_single, 
        #                                   region_feat_single,
        #                                   region_feat_single,
        #                                   key_padding_mask=mask_single)[0]
        #         region_feat_att.append(tmp)
        #     region_feat_att = torch.cat(region_feat_att, dim=0)  # T, batch_size, d
        #     output = frame_feat_ori + self.dropout(region_feat_att)
        #     output = self.norm1(output)
        #     output2 = self.linear2(self.dropout(F.relu(self.linear1(output))))
        #     output = output + self.dropout(output2)
        #     output = self.norm2(output)

        # elif 'attV4' in self.mode:
        #     ############# V4 ##############
        #     # Q: region (nxd)
        #     # K: region (nxd)
        #     # V: region (nxd)
        #     # out: regions of each frame then concat with the corresponding frame (nxd) -> (Nxd)
        #     region_feat_att = []
        #     for i in range(T):
        #         region_feat_single = region_feat[i*box_num : (i+1) * box_num]
        #         mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

        #         tmp = self.multihead_attn(region_feat_single, 
        #                                   region_feat_single,
        #                                   region_feat_single,
        #                                   key_padding_mask=mask_single)[0]
        #         region_feat_att.append(tmp)
            
        #     region_feat_att = torch.cat(region_feat_att, dim=0)  # N, batch_size, d
        #     output = frame_feat + region_feat_att
            
        # elif 'attV5' in self.mode:
        #     ############# V5 ##############
        #     # Q: region (nxd)
        #     # K: region (nxd)
        #     # V: region (nxd)
        #     # out: frame feature described by region (nxd), then Avgpooling (1,d) -> (Txd)
        #     region_feat_att = []
        #     for i in range(T):
        #         frame_feat_single = frame_feat_ori[i].unsqueeze(0)
        #         region_feat_single = region_feat[i*box_num : (i+1) * box_num]
        #         mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

        #         tmp = self.multihead_attn(region_feat_single, 
        #                                   region_feat_single,
        #                                   region_feat_single,
        #                                   key_padding_mask=mask_single)[0]
        #         tmp = self.multihead_attn2(frame_feat_single, 
        #                                    tmp,
        #                                    tmp,
        #                                    key_padding_mask=mask_single)[0]
        #         region_feat_att.append(tmp)
        #     region_feat_att = torch.cat(region_feat_att, dim=0)  # T, batch_size, d
        #     output = frame_feat_ori + region_feat_att
