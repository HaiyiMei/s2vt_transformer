import torch 
from torch import nn
# from models.GuideNet import GuideNet
from models.fusion import Fusion

class Encoder_Transformer(nn.Module):
    def __init__(self, opt):
        super(Encoder_Transformer, self).__init__()
        
        self.trans_encoder = opt["transformer_encoder"]
        self.hidden_dim = opt["dim_hidden"]
        self.video_dim = opt["dim_vid"]
        self.with_box = opt["with_box"]
        self.fusion = opt["fusion"]
        self.only_box = opt["only_box"]
        self.T = 32

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)

        if self.trans_encoder:
            transformer_dim = self.hidden_dim*2 if 'encoder' in str(opt["fusion"]) else self.hidden_dim
            encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4)
            self.trans_encoder = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(transformer_dim))
        if self.fusion:
            self.fusion = Fusion(self.video_dim, self.hidden_dim, mode=opt["fusion"], n_layer=opt["n_layer_fusion"])
        self.dropout = nn.Dropout(0.1)

    def img_embedding(self, input, mask=None):
        input = self.img_embed(self.dropout(input))  # batch_size, T, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input
    
    def region_mask(self, input_box):
        region_mask = input_box.sum(-1).eq(0)  # batch_size x N
        N = input_box.size(1)
        box_num = N // self.T
        zeros = region_mask.reshape(-1, self.T, box_num).sum(-1).eq(0)  # batch_size, T
        for i in range(input_box.size(0)):
            for j in range(self.T):
                if zeros[i, j]:
                    region_mask[i, j*box_num].fill_(False)    
        return region_mask

    def forward(self, input_image, input_box=None):
        '''
        input_image: batch_size x T x image_dim
        input_box: batch_size x N x image_dim
        return input_feature: N x batch_size, hidden_dim
        '''
        # frame_mask = input_image.sum(-1).eq(0) if self.only_box else None  # batch_size x N

        if self.fusion:
            # region_mask = self.region_mask(input_box)
            region_mask = input_box.sum(-1).eq(0)  # batch_size x N
            input_feature = self.fusion(input_image, input_box, mask=region_mask)
            mask = region_mask if input_feature.size(1)==input_box.size(1) else None
        else:
            frame_mask = self.region_mask(input_image) if self.only_box else None  # batch_size x N
            input_image = self.img_embedding(input_image, mask=frame_mask)
            input_feature = input_image
            mask = frame_mask

        # batch_size, T, d
        input_feature = input_feature.transpose(0, 1)  # N, batch_size, d

        # transformer encoder
        if self.trans_encoder:
            input_feature = self.trans_encoder(input_feature, src_key_padding_mask=mask)

        return input_feature, mask
    