import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data
        self.feats_dir = opt["feats_dir"]
        self.dataset = opt["dataset"]
        self.captions_maxnum = 81 if self.dataset == 'MSVD' else 20

        self.img = os.path.join(self.feats_dir, '{}_img'.format(opt["model"]))
        self.box = os.path.join(self.feats_dir, '{}_box'.format(opt["model"]))
        if opt["res_box"]:
            self.box = os.path.join(self.feats_dir, 'resnet_box')
        if opt["rpn"]:
            self.box = os.path.join(self.feats_dir, 'rpn')
        if opt["incep_res_box"]:
            self.box = os.path.join(self.feats_dir, 'incep_res_box')

        # load the json file which contains information about the dataset
        self.captions = json.load(open('data/caption_{}.json'.format(self.dataset)))
        info = json.load(open('data/info_{}.json'.format(self.dataset)))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        self.splits = info['videos']

        self.with_box = opt['fusion']
        if opt["only_box"]:
            self.feats_dir = [self.box]
        elif self.with_box:
            self.feats_dir = [self.img, self.box]
        else:
            self.feats_dir = [self.img]

        list_dir = os.listdir(self.img)
        list_dir = [i[:-4] for i in list_dir]
        self.list_all = [item for item in self.splits[mode] if item in list_dir]

        # load in the sequence data
        self.max_len = opt["max_len"]


    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(torch.load(os.path.join(dir, self.list_all[ix]+'.pth')))
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions[self.list_all[ix]]['final_captions']

###########################################################################
        cap_ix = random.randint(0, len(captions) - 1)
        cap = captions[cap_ix]
        if len(cap) > self.max_len:
            cap = cap[:self.max_len]
            cap[-1] = '<eos>'
        for j, w in enumerate(cap):
            label[j] = self.word_to_ix[w]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1
###########################################################################

        # gts = np.zeros((len(captions), self.max_len))
        # for i, cap in enumerate(captions):
        #     if len(cap) > self.max_len:
        #         cap = cap[:self.max_len]
        #         cap[-1] = '<eos>'
        #     for j, w in enumerate(cap):
        #         gts[i, j] = self.word_to_ix[w]

        # # random select a caption for this video
        # cap_ix = random.randint(0, len(captions) - 1)
        # label = gts[cap_ix]
        # non_zero = (label == 0).nonzero()
        # mask[:int(non_zero[0][0]) + 1] = 1

        if self.dataset == 'MSVD':
            gts_ix = random.sample(range(len(captions)), 5)
            # gts_ = np.zeros((self.captions_maxnum, self.max_len))
            # gts_[:len(gts)] = gts
            # gts = gts_

###########################################################################

        data = {}
        data['img_feats'] = fc_feat[0].type(torch.FloatTensor)
        if self.with_box:
            data['box_feats'] = fc_feat[1].type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        # data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = self.list_all[ix]
        return data

    def __len__(self):
        return len(self.list_all)
