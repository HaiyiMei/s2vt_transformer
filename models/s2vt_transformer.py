import torch.nn as nn 
from models.Decoder_Transformer import Decoder_Transformer

class S2VT_Transformer(nn.Module):
    def __init__(self, opt):
        super(S2VT_Transformer, self).__init__()

        self.hidden_dim = opt["dim_hidden"]
        self.video_dim = opt["dim_vid"]

        if opt["fusion"]:
            from models.Encoder_channel import Encoder_Transformer
        else:
            from models.Encoder_Transformer import Encoder_Transformer
        self.encoder = Encoder_Transformer(opt)
        self.decoder = Decoder_Transformer(opt)

        # self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)

    def forward(self, input_image, input_caption=None, input_box=None, mode='train'):
        '''
        input_image: batch_size x T x image_dim
        input_box: batch_size x N x image_dim
        input_feature: N x batch_size, hidden_dim
        mask: batch_size x N
        input_caption: batch_size x caption_step 
        '''

        input_feature, mask = self.encoder(input_image, input_box)
        seq_probs, seq_preds = self.decoder(input_feature, mask, input_caption, mode)

        return seq_probs, seq_preds