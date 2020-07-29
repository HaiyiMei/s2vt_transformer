import os
import json
import time
import tqdm
import random
import shutil

import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
import torchnet as tnt

import opts
from misc import utils
from data.dataloader import VideoDataset
from models.Decoder_Transformer import Decoder_Transformer
from misc.cocoeval import suppress_stdout_stderr, COCOScorer


def train(loader, model, optimizer, val=False):
    loss_sum = tnt.meter.AverageValueMeter()
    for data in loader:
        torch.cuda.synchronize()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()

        optimizer.zero_grad()
        seq_probs, _ = model(labels, None, labels, mode='train')
        loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

        if not val:
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()

        loss_sum.add(loss.item())
        torch.cuda.synchronize()
    return loss_sum.value()[0]  # loss mean


def main(opt):
    trainset = VideoDataset(opt, 'train')
    validset = VideoDataset(opt, 'val')

    trainloader = DataLoader(trainset, batch_size=opt["batch_size"], shuffle=True)
    validloader = DataLoader(validset, batch_size=opt["batch_size"], shuffle=False)

    gts = utils.convert_data_to_coco_scorer_format(
        json.load(open('data/annotations_{}.json'.format(opt["dataset"]))))

    opt["vocab_size"] = trainset.get_vocab_size()
    vocab = trainset.get_vocab()

    model = Decoder_Transformer(opt)
    model = model.cuda()
    print(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=1e-3)

    if opt["warmup"] != -1:
        from warmup_scheduler import GradualWarmupScheduler
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt["epochs"])
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=opt["warmup"], after_scheduler=scheduler_cosine) # 200


    metrics_valid = pd.DataFrame(columns=['epoch', 'CIDEr'])
    metrics_test = pd.DataFrame(columns=['epoch', 'Bleu_1', 
        'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'])

    print("dataset: %s"%opt["dataset"], 
          "load feature from: %s"%trainset.feats_dir,
          "saving checkpoint to: %s"%opt["save_path"], 
          "start training", 
          "---------------------------", sep="\n")

    mode = 'beam' if opt["beam"] else 'inference'
    top_val = []

    for epoch in range(opt["epochs"]):
        start_time = time.time()
        is_best = False
        
        if opt["warmup"] != -1:
            scheduler_warmup.step()

        model.train()
        train_loss = train(trainloader, model, optimizer)
        # tensorboard writer
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'] , epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss = train(validloader, model, optimizer, val=True)
            print_info = ["epoch: {}/{}".format(epoch+1, opt["epochs"]), 
                          "train_loss: {:.3f}".format(train_loss),
                          "val_loss: {:.3f}".format(valid_loss)]
            writer.add_scalar('Loss/valid', valid_loss, epoch)

            top_val.append(valid_loss)
            is_best = valid_loss <= min(top_val)

            if is_best:
                model_path = os.path.join('data/{}_decoder.pth'.format(opt['dataset']))
                torch.save(model.state_dict(), model_path)

        # print information
        print_info.append('time: {:.2f}s'.format(time.time()-start_time))
        print('  |  '.join(print_info))
        if (epoch+1) % 10 == 0:
            print('tensorboard --logdir={}'.format(os.path.join(os.getcwd(), writer.log_dir)))


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    print(json.dumps(opt, indent=1))

    # set up random seed
    seed = opt['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # set up gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt["gpu"])
    writer = utils.get_writer(opt)

    # loss criterion
    crit = utils.LanguageModelCriterion()
    main(opt)
