import os
import json

import torch
from torch.utils.data import DataLoader

import opts
from misc import utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from data.dataloader import VideoDataset
from models.s2vt import S2VT


def test(model, loader, vocab, scorer, mode='inference'):
    samples = {}
    for data in loader:
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

    model = S2VT(opt)
    model = model.cuda()

    checkpoint_path = os.path.join(
        opt["checkpoint_path"], 'checkpoint', 'model_{}.pth'.format(opt["checkpoint_epoch"]))
    model.load_state_dict(state_dict=
        torch.load(checkpoint_path))

    print("dataset: %s"%opt["dataset"], 
          "load feature from: %s"%testset.feats_dir,
          "load checkpoint from: %s"%opt["checkpoint_path"],
          "start evaluating", sep="\n")

    model.eval()
    with torch.no_grad():
        test_score = test(model, testloader, vocab, test_scorer, mode=mode)
    
    for k, v in test_score.items():
        print('{}: {:.1f}'.format(k, v))


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    # set up gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]

    main(opt)