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
from models.s2vt import S2VT
from data.dataloader import VideoDataset
from misc.cocoeval import suppress_stdout_stderr, COCOScorer


def train(loader, model, optimizer, val=False):
    loss_sum = tnt.meter.AverageValueMeter()
    for data in loader:
        torch.cuda.synchronize()
        img_feats = data['img_feats'].cuda()
        box_feats = data['box_feats'].cuda() if opt["fusion"] else None
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()

        optimizer.zero_grad()
        seq_probs, _ = model(
            frame_feat=img_feats, 
            region_feat=box_feats,  # batch_size*(box_num_per_frame*frame_num)*2048
            input_caption=labels, 
            mode='train')
        loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

        if not val:
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()

        loss_sum.add(loss.item())
        torch.cuda.synchronize()
    return loss_sum.value()[0]  # loss mean


def test(model, loader, vocab, scorer, mode='inference'):
    samples = {}
    for data in loader:
        # forward the model to get loss
        img_feats = data['img_feats'].cuda()
        box_feats = data['box_feats'].cuda() if opt["fusion"] else None
        video_ids = data['video_ids']
      
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            _, seq_preds = model(
                frame_feat=img_feats, 
                region_feat=box_feats,  # batch_size*(box_num_per_frame*frame_num)*2048
                mode=mode)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(samples)
            
    return valid_score, samples


def main(opt):
    trainset = VideoDataset(opt, 'train')
    validset = VideoDataset(opt, 'val')

    trainloader = DataLoader(trainset, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["workers"])
    validloader = DataLoader(validset, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["workers"])

    gts = utils.convert_data_to_coco_scorer_format(
        json.load(open('data/annotations_{}.json'.format(opt["dataset"]))))

    val_scorer = COCOScorer(gts, validset.list_all, CIDEr=True)

    opt["vocab_size"] = trainset.get_vocab_size()
    vocab = trainset.get_vocab()

    model = S2VT(opt)
    model = model.cuda()

    # print(model)

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
          "start training", "---------------------------", sep="\n")

    mode = 'beam' if opt["beam"] else 'inference'
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
            
            if epoch+1 >= opt["eval_start"]:
                val_score, val_sents = test(model, validloader, vocab, val_scorer, mode=mode)
                print_info.append("val_CIDEr: {:.1f}".format(val_score["CIDEr"]))
                writer.add_scalar('CIDEr/valid', val_score["CIDEr"], epoch)

                val_score.update({'epoch': epoch})
                metrics_valid = metrics_valid.append(val_score, ignore_index=True)
                metrics_valid.to_csv(metrics_log_valid, float_format='%.1f', index=False)
                metrics_steady = metrics_valid if epoch<=steady_epoch else \
                                 metrics_valid[metrics_valid['epoch']>steady_epoch]
                top = metrics_steady.sort_values(by=['CIDEr'], ascending=False)[:top_metics]
                top_CIDer = top['CIDEr'].to_list()

                # with open(sents_json + '/valid_%d.json'%(epoch), 'w') as f:
                #     json.dump(val_sents, f)

                is_best = val_score['CIDEr'] >= min(top_CIDer)

            if is_best:
                model_path = os.path.join(opt["save_path"],
                                          'checkpoint_tmp',
                                          'model_%d.pth' % (epoch))
                torch.save(model.state_dict(), model_path)

        with open(loss_log_file, "a") as log_fp:
            log_fp.write("{0},{1},{2}\n".format(
                epoch, train_loss, valid_loss))
        
        # print information
        print_info.append('time: {:.2f}s'.format(time.time()-start_time))
        print('  |  '.join(print_info))
        if (epoch+1) % 10 == 0:
            print('tensorboard --logdir={}'.format(os.path.join(os.getcwd(), writer.log_dir)))

    testset = VideoDataset(opt, 'test')
    testloader = DataLoader(testset, batch_size=opt["batch_size"], shuffle=False)
    test_scorer = COCOScorer(gts, testset.list_all)
    mode = 'beam'
    # select the top result
    top = metrics_steady.sort_values(by=['CIDEr'], ascending=False)[:top_metics]
    top = list(top['epoch'])

    model.eval()
    for epoch in tqdm.tqdm(top, desc='epochs', leave=True):
        model_path = os.path.join(opt["save_path"], 'checkpoint_tmp', 'model_%d.pth' % (epoch))
        model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            test_score, test_sents = test(model, testloader, vocab, test_scorer, mode=mode)

        test_score.update({'epoch': epoch})
        metrics_test = metrics_test.append(test_score, ignore_index=True)
        metrics_test.to_csv(metrics_log_test, float_format='%.1f', index=False)

        with open(sents_json + '/test_%d.json'%(epoch), 'w') as f:
            json.dump(test_sents, f)
        if opt["save_checkpoint"]:
            os.makedirs(
                os.path.join(
                    opt["save_path"], 'checkpoint'), 
                exist_ok=True)
            model_save_path = os.path.join(opt["save_path"], 'checkpoint', 'model_%d.pth' % (epoch))
            shutil.copyfile(model_path, model_save_path)

    shutil.rmtree(os.path.join(opt["save_path"], 'checkpoint_tmp'))

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)

    if os.path.exists(opt["save_path"]):
        end = input(f'{opt["save_path"]} exsits. delete it? ([y]/n)')
        if end == '' or end == 'y':
            shutil.rmtree(opt["save_path"])
        else:
            exit()

    print(json.dumps(opt, indent=1))
    top_metics = 10
    steady_epoch = opt["self_crit_after"] 

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
    sents_json = os.path.join(opt["save_path"], 'sents')
    os.makedirs(sents_json)
    opt_json = os.path.join(opt["save_path"], 'opt_info.json')
    os.makedirs(
        os.path.join(
            opt["save_path"], 'checkpoint_tmp'), 
        exist_ok=True)
    with open(opt_json, 'w') as f:
        json.dump(opt, f)

    # log files
    loss_log_file = os.path.join(opt["save_path"], 'loss.csv')
    metrics_log_valid = os.path.join(opt["save_path"], 'metrics_valid.csv')
    metrics_log_test = os.path.join(opt["save_path"], 'metrics_test.csv')
    with open(loss_log_file, 'w') as log_fp:
        log_fp.write('epoch,train_loss,valid_loss\n')
    writer = utils.get_writer(opt)

    # loss criterion
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    main(opt)
