import numpy as np
import sys
sys.path.append("misc")
import utils
import torch 
from torch import nn
import torch.nn.functional as F
# from models.GCN_model_nobatch import GCN_sim
from models.GCN_model import GCN_sim, GCN_sim_nobatch
from models.GuideNet import GuideNet

class S2VT_Transformer_box(nn.Module):
    def __init__(self, opt):
        super(S2VT_Transformer_box, self).__init__()
        
        self.encoder = opt["transformer_encoder"]
        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.caption_maxlen = opt["max_len"]
        self.video_dim = opt["dim_vid"]
        self.with_box = opt["with_box"]
        self.time_gcn = opt["tg"]
        self.box_gcn = opt["bg"]
        self.guide = opt["guide"]

        # self.word2vec = nn.Embedding(self.word_num, self.hidden_dim)
        self.word2vec = nn.Embedding(self.word_num, 
                                     self.hidden_dim,
                                     padding_idx=0)
        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)

        if self.encoder:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=4)
            self.encoder_frame = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
            self.encoder_region = nn.TransformerEncoder(encoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        self.tgt_mask = self._generate_square_subsequent_mask(self.caption_maxlen).cuda()
        self.pos_encoder = utils.PositionalEncoding(self.hidden_dim)


        self.guide = GuideNet(self.hidden_dim, mode=opt["guide_mode"])

        # self.multihead_attn = nn.MultiheadAttention(self.hidden_dim, 4, dropout=0.1)
        # self.linear1 = nn.Linear(self.hidden_dim, 2048)
        # self.linear2 = nn.Linear(2048, self.hidden_dim)
        # self.norm1 = nn.LayerNorm(self.hidden_dim)
        # self.norm2 = nn.LayerNorm(self.hidden_dim)
        # self.dropout_ff = nn.Dropout(0.1)

        # gcn_sim = GCN_sim_nobatch
        # gcn_sim = GCN_sim
        # if self.time_gcn:
        #     self.time_gcn = gcn_sim(self.video_dim)
        # if self.box_gcn:
        #     self.box_gcn = gcn_sim(self.video_dim)
        # if self.guide:
        #     self.guide = GuideNet(self.video_dim)

        self.dropout = nn.Dropout(0.5)


    def img_embedding(self, input, mask=None):
        input = self.img_embed(self.dropout(input))  # batch_size, T, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input
    

    def forward(self, input_image, input_caption=None, input_box=None, box_num=None, mode='train'):
        '''
        input_image: int Variable, batch_size x T x image_dim
        input_box: int Variable, batch_size x N x image_dim
        input_caption: int Variable, batch_size x caption_step 
        '''

        mask = input_box.sum(-1).eq(0)  # batch_size x N
        # encoding
        # if self.time_gcn:
        #     input_image = self.time_gcn(input_image)
        input_image = self.img_embedding(input_image).transpose(0, 1)
        input_box = self.img_embedding(input_box, mask=mask).transpose(0, 1)

        # transformer encoder
        if self.encoder:
            input_image = self.encoder_frame(input_image)
            input_box = self.encoder_region(input_box, src_key_padding_mask=mask)
        
        input_feature = self.guide(input_image.transpose(0, 1), input_box.transpose(0, 1), mask=mask)

        input_feature = input_feature.transpose(0, 1)  # N, batch_size, d
        
        # tgt = self.multihead_attn(input_box, input_image, input_image)[0]
        # input_feature = input_box + self.dropout_ff(tgt)
        # tgt = self.multihead_attn(input_image, input_box, input_box, key_padding_mask=mask)[0]
        # input_feature = input_image + self.dropout_ff(tgt)
        # input_feature = self.norm1(input_feature)

        # tgt2 = self.linear2(self.dropout(F.relu(self.linear1(input_feature))))
        # input_feature = input_feature + self.dropout_ff(tgt2)
        # input_feature = self.norm2(input_feature)
        
        # decoding
        seq_probs = []
        seq_preds = []
        if mode == 'train':
            input_caption = self.word2vec(input_caption).transpose(0, 1) # encode the words. caption_step, batch_size, hidden_dim
            input_positional = self.pos_encoder(input_caption)
            output = self.decoder(input_positional,  
                                  input_feature,  # transformer is written caption_step, batch_size, hidden_dim  
                                  tgt_mask=self.tgt_mask,
                                  memory_key_padding_mask=mask)  # may should be removed
            output = output[:-1].transpose(0, 1)  # back in batch first
            logits = self.vec2word(self.dropout(output))
            logits = F.log_softmax(logits, dim=-1)
            seq_probs = logits
        else:
            input_caption = torch.ones(input_feature.size(1), 1).long().cuda()  # batch_size, 1, hidden_dim
            input_caption = self.word2vec(input_caption).transpose(0, 1) # encode the words. 1, batch_size, hidden_dim
            for step in range(self.caption_maxlen-1):
                input_positional = self.pos_encoder(input_caption)
                output = self.decoder(input_positional, 
                                      input_feature,
                                      memory_key_padding_mask=mask)
                output = output[-1]  # pick up the last word (btach_size, hidden_dim)

                logits = self.vec2word(self.dropout(output))  # (btach_size, word_num)
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))  # probability (btach_size, 1, word_num)
                if mode == 'sample':
                     preds = torch.multinomial(logits.exp(), 1).cuda()
                     preds = preds.squeeze()
                else:
                    _, preds = torch.max(logits, 1)  # result (batch_size)

                input_caption = torch.cat([input_caption, self.word2vec(preds).unsqueeze(0)], 0)  # step, batch_size, hidden_dim
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask