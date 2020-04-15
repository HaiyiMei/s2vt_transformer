import torch
from torch import nn
import torch.nn.functional as F
# from models.GCN_model_nobatch import GCN_sim
from models.GCN_model import GCN_sim
# from models.Encoder_lstm import Encoder_lstm
from models.fusion_lstm import Fusion


class S2VT_LSTM(nn.Module):
    def __init__(self, opt):
        super(S2VT_LSTM, self).__init__()

        self.word_num = opt["vocab_size"]
        self.hidden_dim = opt["dim_hidden"]
        self.caption_maxlen = opt["max_len"]
        self.video_dim = opt["dim_vid"]
        self.box_dim = opt["dim_box"]
        self.with_box = opt["with_box"]
        self.time_gcn = opt["tg"]
        self.box_gcn = opt["bg"]
        self.fusion = opt["fusion"]
        # self.init_num = self.video_dim+self.box_dim if self.with_box else self.video_dim
        self.init_num = self.video_dim
        if 'concat' in str(opt["fusion"]):
            self.init_num *= 2

        self.word2vec = nn.Embedding(self.word_num, self.hidden_dim)

        self.init_output = nn.Linear(self.init_num, self.hidden_dim)
        self.init_state = nn.Linear(self.init_num, self.hidden_dim)
        self.img_embed = nn.Linear(self.init_num, self.hidden_dim)

        self.vec2word = nn.Linear(self.hidden_dim, self.word_num)
        self.lstm = nn.LSTMCell(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim)

        if self.time_gcn:
            self.time_gcn = GCN_sim(self.video_dim, self.video_dim)
        if self.box_gcn:
            self.box_gcn = GCN_sim(self.box_dim, self.video_dim)
        if self.fusion:
            self.fusion = Fusion(self.video_dim, self.box_dim, self.hidden_dim, mode=opt["fusion"], n_layer=opt["n_layer_fusion"])

        self.dropout = nn.Dropout(0.1)

    def init_hidden(self, input_feature):
        output = self.init_output(input_feature)
        state = self.init_state(input_feature)
        return output, state

    def process_feature(self, input_image, input_gcn):
        batch_size = input_image.size(0)
        if self.time_gcn:
            input_feature = self.time_gcn(input_image)
            return input_feature.mean(1)

        mask = input_gcn.sum(-1).eq(0)
        mask_frame = mask.reshape(batch_size, input_image.size(1), -1)
        mask_frame[:, :, 0] = False
        mask_frame = mask_frame.reshape(batch_size, -1)

        input_feature = self.fusion(input_image, input_gcn, mask, mask_frame)
        input_feature = self.box_gcn(input_feature, mask=mask_frame)

        box_num = (~mask_frame).sum(-1).unsqueeze(-1)  # batch_size x 1
        input_feature = input_feature.sum(1) / box_num.clamp(min=1)

        return input_feature

    # def process_feature(self, input_image, input_gcn):
    #     input_image = torch.stack([self.time_gcn(item, fc=True).mean(0) for item in input_image]) if self.time_gcn \
    #              else torch.stack([item.mean(0) for item in input_image])
    #     if self.with_box:
    #         tmp = []
    #         for item in input_gcn:
    #             item = item[item.sum(-1)!=0]
    #             if len(item)==0:
    #                 tmp.append(torch.zeros(self.box_dim).cuda())
    #             else:
    #                 if self.box_gcn:
    #                     tmp.append(self.box_gcn(item, fc=True).mean(0))
    #                 else:
    #                     tmp.append(item.mean(0))
    #         input_gcn = torch.stack(tmp)
    #         input_feature = torch.cat([input_image, input_gcn], dim=1)
    #     else:
    #         input_feature = input_image
    #     return input_feature


    def forward(self, input_image, input_caption=None, input_box=None, mode='train'):
        '''
        input_image: int Variable, batch_size x image_dim
        input_caption: int Variable, batch_size x (1+caption_step) x 1 (word is idx, so the dim is 1)
        '''

        input_feature = self.process_feature(input_image, input_box)
        
        #encoding
        output, state = self.init_hidden(self.dropout(input_feature))
        input_feature = self.img_embed(self.dropout(input_feature))

        seq_probs = []
        seq_preds = []
        if mode == 'train':
            word_vec = self.word2vec(input_caption)
            for step in range(self.caption_maxlen-1):

                output, state = self.lstm(
                    torch.cat([word_vec[:,step], input_feature], dim=1), 
                    (output, state))

                logits = self.vec2word(self.dropout(output))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
        else:
            current_words = self.word2vec(torch.ones(len(input_feature)).long().cuda())
            for step in range(self.caption_maxlen-1):

                output, state = self.lstm(
                    torch.cat([current_words, input_feature], dim=1), 
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