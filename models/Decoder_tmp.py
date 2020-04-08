import sys
sys.path.append("misc")
import utils
import torch 
from torch import nn
import torch.nn.functional as F


def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


class Decoder_Transformer(nn.Module):
    def __init__(self, opt):
        super(Decoder_Transformer, self).__init__()
        
        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.caption_maxlen = opt["max_len"]

        self.word2vec = nn.Embedding(self.word_num, 
                                     self.hidden_dim,
                                     padding_idx=0)
        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4)
        self.decoder1 = nn.TransformerDecoder(decoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        self.decoder2 = nn.TransformerDecoder(decoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        self.tgt_mask = self._generate_square_subsequent_mask(self.caption_maxlen).cuda()
        self.pos_encoder = utils.PositionalEncoding(self.hidden_dim)

        self.dropout = nn.Dropout(0.5)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, input_image, input_box, input_caption=None, mode='train', beam_size=3):
        '''
        input_feature: N x batch_size, hidden_dim
        mask: batch_size x N
        input_caption: batch_size x caption_step 
        '''

        batch_size = input_image.size(1)
        seq_probs = []
        seq_preds = []

        if mode == 'train':
            input_caption = self.word2vec(input_caption).transpose(0, 1) # encode the words. caption_step, batch_size, hidden_dim
            input_positional = self.pos_encoder(input_caption)
            output = self.decoder1(input_image,  
                                   input_box) # transformer is written caption_step, batch_size, hidden_dim  
            output = self.decoder2(input_positional,  
                                   output, # transformer is written caption_step, batch_size, hidden_dim  
                                   tgt_mask=self.tgt_mask)
            output = output[:-1].transpose(0, 1)  # back in batch first
            logits = self.vec2word(self.dropout(output))
            seq_probs = F.log_softmax(logits, dim=-1)

        else:
            input_caption = torch.ones(1, batch_size).long().cuda()  # batch_size, 1
            input_caption = self.word2vec(input_caption) # encode the words. 1, batch_size, hidden_dim
            for step in range(self.caption_maxlen-1):
                input_positional = self.pos_encoder(input_caption)
                output = self.decoder1(input_image,  
                                       input_box) # transformer is written caption_step, batch_size, hidden_dim  
                output = self.decoder2(input_positional,  
                                       output) # transformer is written caption_step, batch_size, hidden_dim  
                output = output[-1]  # pick up the last word (btach_size, hidden_dim)

                logits = self.vec2word(self.dropout(output))  # (btach_size, word_num)
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))  # probability (btach_size, 1, word_num)
                if mode == 'sample':
                     preds = torch.multinomial(logits.exp(), 1).cuda()
                     preds = preds.squeeze()
                elif mode == 'inference':
                    _, preds = torch.max(logits, 1)  # result (batch_size)

                input_caption = torch.cat([input_caption, self.word2vec(preds).unsqueeze(0)], 0)  # step, batch_size, hidden_dim
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)

        return seq_probs, seq_preds
    
