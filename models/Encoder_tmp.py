import torch 
from torch import nn
import torch.nn.functional as F

class Encoder_Transformer(nn.Module):
    def __init__(self, opt):
        super(Encoder_Transformer, self).__init__()
        
        self.trans_encoder = opt["transformer_encoder"]
        self.hidden_dim = opt["dim_hidden"]
        self.video_dim = opt["dim_vid"]
        self.with_box = opt["with_box"]
        self.fusion = opt["fusion"]
        self.only_box = opt["only_box"]

        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.full_att = nn.Linear(self.hidden_dim, 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=4)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        self.trans_encoder_img = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))

        self.dropout = nn.Dropout(0.5)

    def img_embedding(self, input, mask=None):
        input = self.img_embed(self.dropout(input))  # batch_size, T, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input
    
    def forward(self, input_image, input_box=None):
        '''
        input_image: batch_size x T x image_dim
        input_box: batch_size x N x image_dim
        return input_feature: N x batch_size, hidden_dim
        '''
        batch_size, T, d = input_image.shape
        box_num = input_box.size(1) // T

        input_image = self.img_embedding(input_image, mask=None)
        input_image = input_image.transpose(0, 1)
        input_image = self.trans_encoder_img(input_image)

        mask = input_box.sum(-1).eq(0)  # batch_size x N
        input_box = self.img_embedding(input_box, mask=mask)
        input_box = input_box.transpose(0, 1)
        # input_box = self.trans_encoder(input_box, src_key_padding_mask=mask)

        region_feat = []
        for i in range(T):
            region_feat_single = input_box[i*box_num : (i+1) * box_num]
            mask_single = mask[:, i*box_num : (i+1) * box_num]  # batch_size, box_num

            tmp = self.trans_encoder(region_feat_single, src_key_padding_mask=mask_single)
            region_feat.append(tmp)
        
        input_box = torch.cat(region_feat, dim=0)  # N, batch_size, d
        # for aligned avg
        nonzeros = (~mask).reshape(-1, T, box_num).sum(1)  # batch_size, box_num
        nonzeros = nonzeros.transpose(0, 1).unsqueeze(-1)  # box_num, batch_size, 1
        input_box = input_box.reshape(T, box_num, -1, self.hidden_dim).sum(0) / nonzeros.clamp(min=1)

        # input_box = input_box.transpose(0, 1)
        # att = self.full_att(input_image).squeeze(-1)  # batch_size, T
        # alpha = F.softmax(att, dim=-1)

        # input_box = input_box.reshape(batch_size, T, box_num, -1)
        # input_box = (input_box*alpha.unsqueeze(-1).unsqueeze(-1)).sum(1)  # b_z, box_num, d

        return input_image, input_box
    