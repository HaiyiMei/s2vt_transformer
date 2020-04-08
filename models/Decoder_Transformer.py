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
        self.hidden_dim = self.hidden_dim*2 if 'encoder' in str(opt["fusion"]) else self.hidden_dim
        self.caption_maxlen = opt["max_len"]

        self.word2vec = nn.Embedding(self.word_num, 
                                     self.hidden_dim,
                                     padding_idx=0)
        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        self.tgt_mask = self._generate_square_subsequent_mask(self.caption_maxlen).cuda()
        self.pos_encoder = utils.PositionalEncoding(self.hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, input_feature, mask=None, input_caption=None, mode='train', beam_size=3):
        '''
        input_feature: N x batch_size, hidden_dim
        mask: batch_size x N
        input_caption: batch_size x caption_step 
        '''

        batch_size = input_feature.size(1)
        seq_probs = []
        seq_preds = []

        if mode == 'train':
            input_caption = self.word2vec(input_caption).transpose(0, 1) # encode the words. caption_step, batch_size, hidden_dim
            input_positional = self.pos_encoder(input_caption)
            output = self.decoder(input_positional,  
                                  input_feature, # transformer is written caption_step, batch_size, hidden_dim  
                                  tgt_mask=self.tgt_mask,
                                  memory_key_padding_mask=mask)
            output = output[:-1].transpose(0, 1)  # back in batch first
            logits = self.vec2word(self.dropout(output))
            seq_probs = F.log_softmax(logits, dim=-1)
        elif mode=='beam':
            N, B, D = input_feature.shape
            input_feature = input_feature.unsqueeze(2).expand(-1, -1, beam_size, -1).reshape(N, -1, D)
            assert (input_feature[:, 0]==input_feature[:, 2]).all()
#             if mask is not None:
#                 mask = mask.repeat(beam_size, 1)

            input_words = torch.ones(1, batch_size*beam_size).long().cuda()  # 1, batch_size*beam_size

            for step in range(self.caption_maxlen-1):
                input_caption = self.word2vec(input_words) # encode the words. 1, batch_size*beam_size, hidden_dim
                input_positional = self.pos_encoder(input_caption)
                output = self.decoder(input_positional,
                                      input_feature,
                                      memory_key_padding_mask=mask)
                output = output[-1]  # pick up the last word (btach_size*beam_size, hidden_dim)
                logits = self.vec2word(self.dropout(output))

                if step==0:
                    logits_0 = torch.stack([logits[i*beam_size]  for i in range(batch_size)])
                    top_score, top_words = logits_0.topk(beam_size)  # (batch_size, beam_size)
                    top_score = top_score.view(-1)
                    top_words = top_words.view(-1)

                    input_words = torch.cat([input_words, top_words.unsqueeze(0)], 0)  # step, batch_size*beam_size

                else:
                    scores = top_score.unsqueeze(1).expand(-1, beam_size).reshape(batch_size, -1)  # (batch_size, beam_size^2)
                    top_score, top_words = logits.topk(beam_size)  # (batch_size*beam_size, beam_size)

                    top_score = top_score.view(batch_size, -1)
                    top_words = top_words.view(batch_size, -1)

                    input_words = input_words.view(-1, batch_size, beam_size)
                    input_words = input_words.repeat(1, 1, beam_size)

                    scores += top_score
                    top_score, top_ind = scores.topk(beam_size)

                    input_words = torch.cat([input_words, top_words.unsqueeze(0)], 0)
                    input_words = torch.cat([torch.index_select(input_words[:, i], -1, top_ind[i]) for i in range(batch_size)], dim=1)

                    top_words = input_words[0]

                    top_score = top_score.view(-1)
                    top_words = top_words.view(-1)

            sorted_score, sorted_idx = torch.max(top_score.view(batch_size, beam_size), 1)


            input_words = input_words.reshape(-1, batch_size, beam_size)
            input_words = input_words.transpose(0, 1)
            input_words = torch.stack([input_words[i, :, sorted_idx[i]] for i in range(batch_size)])
            seq_preds = input_words
            # seq_preds, _ = torch.max(input_words, -1)

        else:
            input_caption = torch.ones(1, batch_size).long().cuda()  # 1, batch_size
            input_caption = self.word2vec(input_caption) # encode the words. 1, batch_size, hidden_dim
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
                elif mode == 'inference':
                    _, preds = torch.max(logits, 1)  # result (batch_size)

                input_caption = torch.cat([input_caption, self.word2vec(preds).unsqueeze(0)], 0)  # step, batch_size, hidden_dim
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)

        return seq_probs, seq_preds
    
