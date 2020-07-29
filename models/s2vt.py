import os
import torch
import torch.nn as nn 
from models.Encoder import Encoder
from models.Decoder_Transformer import Decoder_Transformer
from models.Decoder_LSTM import Decoder_LSTM

class S2VT(nn.Module):
    def __init__(self, opt):
        super(S2VT, self).__init__()

        self.encoder = Encoder(opt)
        if opt["decoder"] == 'transformer':
            self.decoder = Decoder_Transformer(opt)
        elif opt["decoder"] == 'lstm':
            self.decoder = Decoder_LSTM(opt)

        # decoder_state_dict = 'data/{}_decoder.pth'.format(opt["dataset"])
        # if os.path.exists(decoder_state_dict):
        #     state_dict = torch.load(decoder_state_dict)
        #     self.decoder.load_state_dict(state_dict)



    def forward(self, frame_feat, region_feat=None, input_caption=None, mode='train'):
        '''
        input_image: batch_size x T x image_dim
        input_box: batch_size x N x image_dim
        input_feature: N x batch_size, hidden_dim
        mask: batch_size x N
        input_caption: batch_size x caption_step 
        '''

        encoder_out, mask = self.encoder(frame_feat, region_feat)
        seq_probs, seq_preds = self.decoder(encoder_out, mask, input_caption, mode)

        return seq_probs, seq_preds
