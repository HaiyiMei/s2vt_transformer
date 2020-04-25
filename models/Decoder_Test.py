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
        self.k = 5  # beam_size
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
    
    def decoder_step(self, encoder_feats, input_caption, tgt_mask=None):
        input_caption = self.word2vec(input_caption) # encode the words. caption_step, batch_size, hidden_dim
        input_positional = self.pos_encoder(input_caption)

        output = self.decoder(input_positional,  
                              encoder_feats,
                              tgt_mask=tgt_mask)
        logits = self.vec2word(self.dropout(output))
        log_softmax_output = F.log_softmax(logits, dim=-1)  # caption_step, batch_size, word_num 
        return log_softmax_output
    
    def forward(self, encoder_out, input_caption=None, mode='train'):
        '''
        encoder_out: T x batch_size, hidden_dim
        input_caption: batch_size x caption_step 
        '''

        batch_size = encoder_out.size(1)
        seq_probs = []
        seq_preds = []

        if mode == 'train':
            input_caption = input_caption.transpose(0, 1)
            tgt_mask = self.tgt_mask.to(encoder_out.device)
            log_softmax_output = self.decoder_step(encoder_out, input_caption, tgt_mask)
            seq_probs = log_softmax_output[:-1].transpose(0, 1)  # back in batch first

        elif mode=='beam':
            self.pos_index = (torch.LongTensor(range(batch_size)) * self.k).view(-1, 1).cuda()
            encoder_out = encoder_out.unsqueeze(2).expand(-1, -1, self.k, -1).reshape(encoder_out.size(0), batch_size*self.k, -1)
            assert (encoder_out[:, 0] == encoder_out[:, 1]).all()

            sequence_scores = torch.Tensor(batch_size*self.k, 1)
            sequence_scores.fill_(-float('Inf'))
            sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(batch_size)]), 0.0)
            sequence_scores = sequence_scores.cuda()

            input_caption = torch.ones(1, batch_size*self.k).long().cuda()  # 1, batch_size
            probs = torch.zeros(batch_size*self.k, 1, self.word_num).cuda()  # batch_size, 1, word_num

            # stored_scores = list()
            # stored_predecessors = list()
            # stored_emitted_symbols = list()

            for step in range(self.caption_maxlen-1):
                log_softmax_output = self.decoder_step(encoder_out, input_caption)[-1]  # batch_size * k, word_num 

                sequence_scores = _inflate(sequence_scores, self.word_num, 1)
                sequence_scores += log_softmax_output
                scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

                output_word = (candidates % self.word_num).view(batch_size * self.k, 1)  # batch_size * k, 1
                sequence_scores = scores.view(batch_size * self.k, 1)
                predecessors = (candidates / self.word_num + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)

                log_softmax_output = log_softmax_output.index_select(0, predecessors.squeeze())

                input_caption = input_caption.index_select(1, predecessors.squeeze())
                probs = probs.index_select(0, predecessors.squeeze())
                input_caption = torch.cat([input_caption, output_word.transpose(0, 1)], 0)  # step, batch_size
                probs = torch.cat([probs, log_softmax_output.unsqueeze(1)], 1)  # batch_size, step, word_num

#############################################################################################################
            #     stored_scores.append(sequence_scores.clone())
            #     stored_predecessors.append(predecessors)
            #     stored_emitted_symbols.append(output_word)

            #     eos_indices = output_word.eq(self.EOS)
            #     if eos_indices.any():
            #         sequence_scores.masked_fill_(eos_indices, -float('inf'))

            # s, l, p = self._backtrack(stored_predecessors, stored_emitted_symbols,
            #                           stored_scores, batch_size)
            # seq_preds = torch.stack(p)
            # seq_preds = seq_preds[:,:,0,0].transpose(0, 1)
#############################################################################################################

            input_caption = input_caption.index_select(1, self.pos_index.squeeze())
            probs = probs.index_select(0, self.pos_index.squeeze())
            seq_preds = input_caption[1:].transpose(0, 1)
            seq_probs = probs[:, 1:]

        else:
            input_caption = torch.ones(1, batch_size).long().cuda()  # 1, batch_size
            for step in range(self.caption_maxlen-1):
                log_softmax_output = self.decoder_step(encoder_out, input_caption)[-1]
                seq_probs.append(log_softmax_output.unsqueeze(1))  # probability (btach_size, 1, word_num)
                if mode == 'sample':
                     preds = torch.multinomial(log_softmax_output.exp(), 1)
                     preds = preds.squeeze(1)
                elif mode == 'inference':
                    _, preds = torch.max(log_softmax_output, 1)  # result (batch_size)

                input_caption = torch.cat([input_caption, preds.unsqueeze(0)], 0)  # step, batch_size
            seq_probs = torch.cat(seq_probs, 1)  # batch_size, maxlen, word_num
            seq_preds = input_caption[1:].transpose(0, 1)

        return seq_probs, seq_preds

    # def _backtrack(self, predecessors, symbols, scores, b):
    #     p = list()
    #     l = [[self.caption_maxlen] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
    #                                                             # Similar to `h_n`

    #     sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
    #     s = sorted_score.clone()

    #     batch_eos_found = [0] * b   # the number of EOS found
    #                                 # in the backward loop below for each batch

    #     t = self.caption_maxlen - 1
    #     t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
    #     while t >= 0:
    #         current_symbol = symbols[t].index_select(0, t_predecessors)
    #         t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()
    #         eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
    #         if eos_indices.dim() > 0:
    #             for i in range(eos_indices.size(0)-1, -1, -1):
    #                 idx = eos_indices[i]
    #                 b_idx = int(idx[0] / self.k)
    #                 res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
    #                 batch_eos_found[b_idx] += 1
    #                 res_idx = b_idx * self.k + res_k_idx

    #                 t_predecessors[res_idx] = predecessors[t][idx[0]]
    #                 current_symbol[res_idx, :] = symbols[t][idx[0]]
    #                 s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
    #                 l[b_idx][res_k_idx] = t + 1

    #         p.append(current_symbol)

    #         t -= 1

    #     s, re_sorted_idx = s.topk(self.k)
    #     for b_idx in range(b):
    #         l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

    #     re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

    #     p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
    #     s = s.data

    #     return s, l, p