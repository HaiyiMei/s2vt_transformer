import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# from models.GCN_model_nobatch import GCN_sim
from models.GCN_model import GCN_sim, GCN_sim_nobatch
from models.GuideNet import GuideNet


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, encoder_out, decoder_hidden, mask=None):
        # encoder_out: batch_size, n, d
        # decoder_hidden: batch_size, d
        att1 = self.encoder_attention(encoder_out)
        att2 = self.decoder_attention(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(-1)  # batch_size, n
        if mask is not None:
            att.masked_fill_(mask, -1e9)
        alpha = self.softmax(att)

        attention_weighted_encoding = (encoder_out*alpha.unsqueeze(2)).sum(1)
        # return attention_weighted_encoding, alpha.squeeze(0)
        return attention_weighted_encoding, alpha


class S2VT_Attention(nn.Module):
    def __init__(self, opt):
        super(S2VT_Attention, self).__init__()
        
        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.caption_maxlen = opt["max_len"]
        self.video_dim = opt["dim_vid"]
        self.with_box = opt["with_box"]
        self.time_gcn = opt["tg"]
        self.box_gcn = opt["bg"]
        self.guide = opt["guide"]
        self.only_box = opt["only_box"]

        # init_dim = self.hidden_dim*2 if self.with_box and not self.guide else self.hidden_dim

        self.word2vec = nn.Embedding(self.word_num, self.hidden_dim)
        self.init_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.init_state = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.img_embed = nn.Linear(self.video_dim, self.hidden_dim)
        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)
        self.lstm = nn.LSTMCell(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim)

        if self.with_box:
            self.fc = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        if self.time_gcn:
            self.time_gcn = GCN_sim(self.hidden_dim)
        if self.box_gcn:
            self.box_gcn = GCN_sim(self.hidden_dim)
        if self.guide:
            self.guide = GuideNet(self.hidden_dim, mode=opt["guide_mode"])

        self.attention = Attention(self.hidden_dim, self.hidden_dim, self.hidden_dim//2)
        self.dropout = nn.Dropout(0.5)

    def init_hidden(self, input_image, mask=None):
        if mask is not None:
            box_num = (~mask).sum(-1).unsqueeze(-1)  # batch_size x N
            input_feature = input_image.sum(1) / box_num.clamp(min=1)
        else:
            input_feature = input_image.mean(1)
        input_feature = self.dropout(input_feature)
        output = self.init_output(input_feature)
        state = self.init_state(input_feature)
        return output, state

    def img_embedding(self, input, mask=None):
        input = self.img_embed(self.dropout(input))  # batch_size, T, d
        if mask is not None:
            input.masked_fill_(mask.unsqueeze(-1).expand(input.shape), 0)
        return input

    def forward(self, input_image, input_caption=None, input_box=None, mode='train'):
        '''
        input_image: int Variable, batch_size x N x image_dim
        input_caption: int Variable, batch_size x (1+caption_step) x 1 (word is idx, so the dim is 1)
        '''

        #encoding
        mask = input_image.sum(-1).eq(0) if self.only_box else None # batch_size x N
        input_image = self.img_embedding(input_image, mask=mask)
        if self.time_gcn:
            input_image = self.time_gcn(input_image, mask=mask)

        output, state = self.init_hidden(input_image, mask=mask)
        
        #decoding
        seq_probs = []
        seq_preds = []
        if mode == 'train':
            word_vec = self.word2vec(input_caption)
            for step in range(self.caption_maxlen-1):
                attention_feat, alpha = self.attention(input_image, state, mask)
                output, state = self.lstm(
                    torch.cat([word_vec[:, step], attention_feat], dim=1), 
                    (output, state))

                logits = self.vec2word(self.dropout(output))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)

        else:
            current_words = self.word2vec(torch.ones(len(input_image)).long().cuda())
            for step in range(self.caption_maxlen-1):
                attention_feat, alpha = self.attention(input_image, state, mask)

                # without Schedule Sample
                output, state = self.lstm(
                    torch.cat([current_words, attention_feat], dim=1), 
                    (output, state))
                
                logits = self.vec2word(self.dropout(output))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.word2vec(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds
    