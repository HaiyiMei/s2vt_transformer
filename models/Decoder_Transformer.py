import sys
sys.path.append("misc")
import utils
import torch 
import torch.nn as nn 
import torch.nn.functional as F


def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)


class Decoder_Transformer(nn.Module):
    def __init__(self, opt):
        super(Decoder_Transformer, self).__init__()
        
        self.SOS = 1
        self.EOS = 0
        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.hidden_dim = self.hidden_dim*2 if 'concat' in str(opt["fusion"]) else self.hidden_dim
        self.caption_maxlen = opt["max_len"]

        self.word2vec = nn.Embedding(self.word_num, 
                                     self.hidden_dim,
                                     padding_idx=0)
        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, opt["n_layer"], nn.LayerNorm(self.hidden_dim))
        self.tgt_mask = self._generate_square_subsequent_mask(self.caption_maxlen)
        self.pos_encoder = utils.PositionalEncoding(self.hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def decoder_step(self, encoder_feats, mask, input_caption, tgt_mask=None):
        input_caption = self.word2vec(input_caption) # encode the words. caption_step, batch_size, hidden_dim
        input_positional = self.pos_encoder(input_caption)

        output = self.decoder(input_positional,  
                              encoder_feats,
                              tgt_mask=tgt_mask,
                              memory_key_padding_mask=mask)
        logits = self.vec2word(self.dropout(output))
        log_softmax_output = F.log_softmax(logits, dim=-1)  # caption_step, batch_size, word_num 
        return log_softmax_output
    
    def forward(self, encoder_feats, mask=None, input_caption=None, mode='train', beam_size=3):
        '''
        encoder_feats: N x batch_size, hidden_dim
        mask: batch_size x N
        input_caption: batch_size x caption_step 
        '''

        batch_size = encoder_feats.size(1)
        seq_probs = []
        seq_preds = []

        if mode == 'train':
            input_caption = input_caption.transpose(0, 1)
            log_softmax_output = self.decoder_step(encoder_feats, mask, input_caption, self.tgt_mask.to(encoder_feats.device))
            seq_probs = log_softmax_output[:-1].transpose(0, 1)  # back in batch first

        elif mode=='beam':
            self.k = beam_size
            self.pos_index = (torch.LongTensor(range(batch_size)) * self.k).view(-1, 1).cuda()
            encoder_feats = encoder_feats.unsqueeze(2).expand(-1, -1, self.k, -1).reshape(encoder_feats.size(0), batch_size*self.k, -1)
            assert (encoder_feats[:, 0] == encoder_feats[:, 1]).all()

            sequence_scores = torch.Tensor(batch_size*self.k, 1)
            sequence_scores.fill_(-float('Inf'))
            sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(batch_size)]), 0.0)
            sequence_scores = sequence_scores.cuda()

            input_caption = torch.ones(1, batch_size*self.k).long().cuda()  # 1, batch_size

            for step in range(self.caption_maxlen-1):
                log_softmax_output = self.decoder_step(encoder_feats, mask, input_caption)[-1]  # batch_size * k, word_num 

                sequence_scores = _inflate(sequence_scores, self.word_num, 1)
                sequence_scores += log_softmax_output

                scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

                input_word = (candidates % self.word_num).view(batch_size * self.k, 1)
                sequence_scores = scores.view(batch_size * self.k, 1)

                predecessors = (candidates / self.word_num + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)

                input_caption = input_caption.index_select(1, predecessors.squeeze())
                log_softmax_output = log_softmax_output.index_select(0, predecessors.squeeze())

                seq_probs.append(log_softmax_output.unsqueeze(1))  # probability (btach_size * k, 1, word_num)
                input_caption = torch.cat([input_caption, input_word.transpose(0, 1)], 0)  # step, batch_size

                # eos_indices = input_word.eq(self.EOS)
                # if eos_indices.nonzero().dim() > 0:
                #     sequence_scores.masked_fill_(eos_indices, -float('inf'))

            seq_probs = torch.cat(seq_probs, 1).index_select(0, self.pos_index.squeeze())
            input_caption = input_caption.index_select(1, self.pos_index.squeeze())
            seq_preds = input_caption[1:].transpose(0, 1)

        else:
            input_caption = torch.ones(1, batch_size).long().cuda()  # 1, batch_size
            for step in range(self.caption_maxlen-1):
                log_softmax_output = self.decoder_step(encoder_feats, mask, input_caption)[-1]
                seq_probs.append(log_softmax_output.unsqueeze(1))  # probability (btach_size, 1, word_num)
                if mode == 'sample':
                     preds = torch.multinomial(log_softmax_output.data.exp().cpu(), 1).cuda()
                     preds = preds.squeeze()
                elif mode == 'inference':
                    _, preds = torch.max(log_softmax_output, 1)  # result (batch_size)

                input_caption = torch.cat([input_caption, preds.unsqueeze(0)], 0)  # step, batch_size
            seq_probs = torch.cat(seq_probs, 1)  # batch_size, maxlen, word_num
            seq_preds = input_caption[1:].transpose(0, 1)

        return seq_probs, seq_preds
