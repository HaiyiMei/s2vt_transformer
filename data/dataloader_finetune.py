import json
import glob
from PIL import Image

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
        self.crop_size = 256
        self.sample_len = opt["sample_len"]

        self.frame_dir = os.path.join(self.feats_dir.rsplit('/', 1)[0], 'frames')
        self.box = os.path.join(self.feats_dir, '{}_box'.format(opt["model"]))
        if opt["rpn"]:
            self.box = os.path.join(self.feats_dir, 'rpn_box')
        if opt["res_box"]:
            self.box = os.path.join(self.feats_dir, 'resnet_box')

        # load the json file which contains information about the dataset
        self.captions = json.load(open('data/caption_{}.json'.format(self.dataset)))
        info = json.load(open('data/info_{}.json'.format(self.dataset)))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        self.splits = info['videos']

        self.with_box = opt['fusion']

        list_dir = os.listdir(self.box)
        list_dir = [i[:-4] for i in list_dir]
        self.list_all = [item for item in self.splits[mode] if item in list_dir]

        # load in the sequence data
        self.max_len = opt["max_len"]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Scale(256),
            #if use TSM full_size, replace 224 with 256
            transforms.CenterCrop(self.crop_size), 
            transforms.ToTensor(),
            self.normalize,
        ])


    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """

        frames = os.path.join(self.frame_dir, self.list_all[ix])
        frames = glob.glob(frames+'/*.jpg')
        frames.sort()
 
        frame_indices = np.linspace(0, len(frames)-1, self.sample_len, dtype=np.int)
        imgs = [frames[idx] for idx in frame_indices]
        imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs = [self.transform(img) for img in imgs] # T*3*224*224
        imgs = torch.stack(imgs, dim=0)

        if self.with_box:
            box_feats = torch.load(os.path.join(self.box, self.list_all[ix]+'.pth'))

        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions[self.list_all[ix]]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        if self.dataset == 'MSVD':
            gts_ix = random.sample(range(len(captions)), 5)
            gts = gts[gts_ix]

        data = {}
        data['img_feats'] = imgs.type(torch.FloatTensor)
        if self.with_box:
            data['box_feats'] = box_feats.type(torch.FloatTensor)

            # feat = torch.zeros(320, 2048)
            # feat[:len(fc_feat[1])] = fc_feat[1]
            # data['box_feats'] = feat
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = self.list_all[ix]
        return data

    def __len__(self):
        return len(self.list_all)