import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.GCN_model import GCN_sim
from models.GuideNet import GuideNet

class S2VT(nn.Module):
    def __init__(self, opt):
        super(S2VT, self).__init__()
        
        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.caption_maxlen = opt["max_len"]
        self.video_dim = opt["dim_vid"]
        self.with_box = opt["with_box"]
        self.time_gcn = opt["tg"]
        self.box_gcn = opt["bg"]
        self.guide = opt["guide"]
        self.only_box = opt["only_box"]

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.lstm = nn.LSTMCell(input_size=self.hidden_dim*2, 
                                hidden_size=self.hidden_dim)

        if self.with_box:
            self.fc = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        if self.time_gcn:
            self.time_gcn = GCN_sim(self.hidden_dim)
        if self.box_gcn:
            self.box_gcn = GCN_sim(self.hidden_dim)
        if self.guide:
            self.guide = GuideNet(self.hidden_dim, mode=opt["guide_mode"])

        self.dropout = nn.Dropout(0.5)

    def img_embedding(self, input, mask=None):
        input = self.img_embed(self.dropout(input))  # batch_size, T, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input

    def forward(self, input_image, input_box=None):
        '''
        input_image: int Variable, batch_size x N x image_dim
        input_caption: int Variable, batch_size x (1+caption_step) x 1 (word is idx, so the dim is 1)
        return input_feature: batch_size x hidden_dim
        '''
        frame_mask = input_image.sum(-1).eq(0) if self.only_box else None # batch_size x N
        input_image = self.img_embedding(input_image, mask=frame_mask)
        if self.time_gcn:
            input_image = self.time_gcn(input_image, mask=frame_mask)
        
        if self.guide:
            region_mask = input_box.sum(-1).eq(0)  # batch_size x N
            input_box = self.img_embedding(input_box, mask=region_mask)
            input_feature = self.guide(input_image, input_box, mask=region_mask)

            box_num = (~region_mask).sum(-1).unsqueeze(-1)  # batch_size x N
            input_feature = input_feature.sum(1) / box_num.clamp(min=1)
        elif self.with_box:
            region_mask = input_box.sum(-1).eq(0)  # batch_size x N
            input_box = self.img_embedding(input_box, mask=region_mask)
            if self.box_gcn:
                input_box = self.box_gcn(input_box, mask=region_mask)

            box_num = (~region_mask).sum(-1).unsqueeze(-1)  # batch_size x N
            input_box = input_box.sum(1) / box_num.clamp(min=1)
            input_feature = torch.cat([input_image.mean(1), input_box], dim=1)
            input_feature = self.fc(input_feature)
        else:
            # OnlyBox
            if frame_mask is not None:
                box_num = (~frame_mask).sum(-1).unsqueeze(-1)  # batch_size x N
                input_feature = input_image.sum(1) / box_num.clamp(min=1)
            # OnlyFrame
            else:
                input_feature = input_image.mean(1)
        
        return input_feature
