import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import os
from torch.utils.tensorboard import SummaryWriter


def get_writer(opt):
    tensorboard_path = os.path.join(opt["save_path"], 'tensorboard_log')
    os.makedirs(tensorboard_path, exist_ok=True)
    # writer = SummaryWriter(opt["save_path"])
    writer = SummaryWriter(tensorboard_path)
    writer.add_text('warmup', str(opt["warmup"]))
    writer.add_text('with_box', str(opt["with_box"]))
    writer.add_text('only_box', str(opt["only_box"]))
    writer.add_text('attention', str(opt["attention"]))
    writer.add_text('frame gcn', str(opt["tg"]))
    writer.add_text('region gcn', str(opt["bg"]))
    writer.add_text('transformer encoder', str(opt["transformer_encoder"]))
    writer.add_text('transformer decoder', str(opt["transformer"]))
    writer.add_text('encoder/decoder layer number', str(opt["n_layer"]))
    writer.add_text('fusion', str(opt["fusion"]))
    
    return writer


# def angle_defn(pos, i, d_model_size):
#     angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
#     return pos * angle_rates


# def positional_encoding(position, d_model_size, dtype):
#     # create the sinusoidal pattern for the positional encoding
#     # return position, d
#     angle_rads = angle_defn(
#         torch.arange(position, dtype=dtype).unsqueeze(1),
#         torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
#         d_model_size,
#     )

#     sines = torch.sin(angle_rads[:, 0::2])
#     cosines = torch.cos(angle_rads[:, 1::2])

#     pos_encoding = torch.cat([sines, cosines], dim=-1)
#     return pos_encoding


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)  # probability for selected words
        print(input.min())
        
        input = input.reshape(-1)  # batch_size, max_len
        reward = reward.reshape(-1)
        mask = (seq>0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        self.loss_fn = nn.NLLLoss(reduction='none')

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output
    
def convert_data_to_coco_scorer_format(annotations_json):
    from pandas import json_normalize
    data_frame = json_normalize(annotations_json['sentences'])
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts