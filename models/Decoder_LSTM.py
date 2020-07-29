import torch
from torch import nn
import torch.nn.functional as F

def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()

        # self.encoder_fc = nn.Linear(encoder_dim, attention_dim)
        # self.word_fc = nn.Linear(hidden_dim, attention_dim)
        self.att = nn.Linear(encoder_dim+hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, encoder_feat, word_feat):
        '''
        encoder_feat: T, batch_size, d
        word_feat: batch_size, d
        return attention_weighted_frame: batch_size, d
        '''
        # att1 = self.encoder_fc(encoder_feat)
        # att2 = self.word_fc(word_feat).unsqueeze(0)
        # att = self.full_att(self.relu(att1 + att2)).squeeze(-1)  # T, batch_size
        ###########################
        word_feat = word_feat.unsqueeze(0).repeat(encoder_feat.size(0), 1, 1)
        att = self.full_att(torch.tanh(self.att(
            torch.cat([encoder_feat, word_feat], dim=-1)
        ))).squeeze(-1)
        alpha = self.softmax(att)  # T, batch_size

        attention_weighted_frame = (encoder_feat*alpha.unsqueeze(-1)).sum(0)
        return attention_weighted_frame

class Decoder_LSTM(nn.Module):
    def __init__(self, opt):
        super(Decoder_LSTM, self).__init__()

        self.SOS = 1
        self.EOS = 0
        self.k = 5  # beam_size
        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.encoder_dim = self.hidden_dim*2 if 'concat' in str(opt["fusion"]) else self.hidden_dim
        self.caption_maxlen = opt["max_len"]

        self.word2vec = nn.Embedding(self.word_num, self.hidden_dim)

        self.init_output = nn.Linear(self.encoder_dim, self.hidden_dim)
        self.init_state = nn.Linear(self.encoder_dim, self.hidden_dim)

        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)
        self.lstm = nn.LSTMCell(input_size=self.hidden_dim+self.encoder_dim, hidden_size=self.hidden_dim)
        self.attention = Attention(self.encoder_dim, self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def init_hidden(self, encoder_out):
        '''
        encoder_out: T x batch_size, hidden_dim
        output, state: batch_size x hidden_dim 
        '''
        encoder_out = encoder_out.mean(0)
        output = self.init_output(encoder_out)
        state = self.init_state(encoder_out)
        return output, state

    def forward(self, encoder_out, mask, input_caption=None, mode='train'):
        '''
        encoder_out: T x batch_size, hidden_dim
        input_caption: batch_size x caption_step 
        '''
        seq_probs = []
        seq_preds = []
        batch_size = encoder_out.size(1)

        output, state = self.init_hidden(self.dropout(encoder_out))

        if mode == 'train':
            word_vec = self.word2vec(input_caption)  # batch_size, caption_step, hidden_dim
            for step in range(self.caption_maxlen-1):
                input_word = word_vec[:, step]
                attention_encoder_out = self.attention(encoder_out, input_word)
                output, state = self.lstm(
                    torch.cat([input_word, attention_encoder_out], dim=1), 
                    (output, state))
                logits = self.vec2word(self.dropout(output))
                log_softmax_output = F.log_softmax(logits, dim=1)
                seq_probs.append(log_softmax_output.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
        elif mode=='beam':
            self.pos_index = (torch.LongTensor(range(batch_size)) * self.k).view(-1, 1).cuda()
            encoder_out = encoder_out.unsqueeze(2).expand(-1, -1, self.k, -1).reshape(encoder_out.size(0), batch_size*self.k, -1)
            output = output.unsqueeze(1).expand(-1, self.k, -1).reshape(batch_size*self.k, -1)
            state = state.unsqueeze(1).expand(-1, self.k, -1).reshape(batch_size*self.k, -1)
            assert (encoder_out[:, 0] == encoder_out[:, 1]).all()
            assert (output[0] == output[1]).all()
            assert (state[0] == state[1]).all()

            sequence_scores = torch.Tensor(batch_size*self.k, 1)
            sequence_scores.fill_(-float('Inf'))
            sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(batch_size)]), 0.0)
            sequence_scores = sequence_scores.cuda()

            input_caption = torch.ones(1, batch_size*self.k).long().cuda()  # 1, batch_size
            probs = torch.zeros(batch_size*self.k, 1, self.word_num).cuda()  # batch_size, 1, word_num
            for step in range(self.caption_maxlen-1):
                input_word = self.word2vec(input_caption[step])
                attention_encoder_out = self.attention(encoder_out, input_word)
                output, state = self.lstm(
                    torch.cat([input_word, attention_encoder_out], dim=1), 
                    (output, state))
                logits = self.vec2word(self.dropout(output))
                log_softmax_output = F.log_softmax(logits, dim=1)

                sequence_scores = _inflate(sequence_scores, self.word_num, 1)
                sequence_scores += log_softmax_output
                scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

                output_word = (candidates % self.word_num).view(batch_size * self.k, 1)
                sequence_scores = scores.view(batch_size * self.k, 1)
                predecessors = (candidates / self.word_num + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)

                output = output.index_select(0, predecessors.squeeze())
                state = state.index_select(0, predecessors.squeeze())
                log_softmax_output = log_softmax_output.index_select(0, predecessors.squeeze())

                input_caption = input_caption.index_select(1, predecessors.squeeze())
                probs = probs.index_select(0, predecessors.squeeze())
                input_caption = torch.cat([input_caption, output_word.transpose(0, 1)], 0)  # step, batch_size
                probs = torch.cat([probs, log_softmax_output.unsqueeze(1)], 1)  # batch_size, step, word_num

                # eos_indices = input_word.eq(self.EOS)
                # if eos_indices.nonzero().dim() > 0:
                #     sequence_scores.masked_fill_(eos_indices, -float('inf'))

            input_caption = input_caption.index_select(1, self.pos_index.squeeze())
            probs = probs.index_select(0, self.pos_index.squeeze())
            seq_preds = input_caption[1:].transpose(0, 1)
            seq_probs = probs[:, 1:]
        else:
            input_word = self.word2vec(torch.ones(batch_size).long().cuda())  # batch_size, hidden_dim
            for step in range(self.caption_maxlen-1):
                attention_encoder_out = self.attention(encoder_out, input_word)
                output, state = self.lstm(
                    torch.cat([input_word, attention_encoder_out], dim=1), 
                    (output, state))
                logits = self.vec2word(self.dropout(output))
                log_softmax_output = F.log_softmax(logits, dim=1)
                seq_probs.append(log_softmax_output.unsqueeze(1))
                if mode == 'sample':
                    preds = torch.multinomial(log_softmax_output.data.exp().cpu(), 1).cuda()
                    preds = preds.squeeze()
                elif mode == 'inference':
                    _, preds = torch.max(log_softmax_output, 1)  # result (batch_size)
                input_word = self.word2vec(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds