import os
import json
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

import opts
from misc import utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from data.dataloader_finetune import VideoDataset
from models.finetune_model import finetune_model


def test(model, loader, vocab, scorer, mode='inference'):
    samples = {}
    for data in tqdm.tqdm(loader):
        # forward the model to get loss
        img_feats = data['img_feats'].cuda()
        video_ids = data['video_ids']
        if opt["with_box"] or opt["fusion"]:
            box_feats = data['box_feats'].cuda()
        else: 
            box_feats = None
      
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            _, seq_preds = model(
                input_image=img_feats, 
                input_box=box_feats, 
                mode=mode)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(samples)
            
    return valid_score


def main(opt):
    testset = VideoDataset(opt, 'test')
    testloader = DataLoader(testset, batch_size=opt["batch_size"], shuffle=False)

    gts = utils.convert_data_to_coco_scorer_format(
        json.load(open('data/annotations_{}.json'.format(opt["dataset"]))))

    test_scorer = COCOScorer(gts, testset.list_all)

    opt["vocab_size"] = testset.get_vocab_size()
    vocab = testset.get_vocab()
    mode = 'beam' if opt["beam"] else 'inference'

    model = finetune_model(opt, evaluate=True)
    model = model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    checkpoint_path = os.path.join(
        opt["save_path"], 'checkpoint_finetune')
    if opt["checkpoint_epoch"] is not None:
        checkpoints = [os.path.join(checkpoint_path, 'model_{}.pth'.format(opt["checkpoint_epoch"]))]
    else:
        checkpoints = os.listdir(checkpoint_path)
        checkpoints = [os.path.join(checkpoint_path, path) for path in checkpoints]

    print("dataset: %s"%opt["dataset"], 
          "load feature from: %s"%testset.feats_dir,
          "load checkpoint from: %s"%opt["save_path"],
          "start evaluating", sep="\n")

    metrics_test = pd.DataFrame(columns=['epoch', 'Bleu_1', 
        'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'])
    metrics_log_test = os.path.join(opt["save_path"], 'metrics_eval.csv')
    
    for checkpoint in tqdm.tqdm(checkpoints):
        model.load_state_dict(state_dict=
            torch.load(checkpoint))

        with torch.no_grad():
            test_score = test(model, testloader, vocab, test_scorer, mode=mode)
    
        test_score.update({'epoch': checkpoint.split('/')[-1][6:9]})
        metrics_test = metrics_test.append(test_score, ignore_index=True)
        metrics_test.to_csv(metrics_log_test, float_format='%.1f', index=False)

    # for k, v in test_score.items():
    #     print('{}: {:.1f}'.format(k, v))


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    # set up gpu

    main(opt)