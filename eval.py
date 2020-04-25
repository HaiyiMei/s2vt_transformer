import os
import json
import pandas as pd

import torch
from torch.utils.data import DataLoader
import tqdm

import opts
from misc import utils
from data.dataloader import VideoDataset
from models.s2vt import S2VT
from misc.cocoeval import suppress_stdout_stderr, COCOScorer


def test(model, loader, vocab, scorer, mode='inference'):
    samples = {}
    frame_weights = {}
    region_weights = {}
    for data in tqdm.tqdm(loader, desc='step', leave=False):
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
        
    #     frame_weight = torch.load('generated_weight/frame/tmp.pth').cpu()
    #     region_weight = torch.load('generated_weight/region/tmp.pth').cpu()
    #     os.remove('generated_weight/frame/tmp.pth')
    #     os.remove('generated_weight/region/tmp.pth')
    #     batch_size = frame_weight.size(0)
    #     for i in range(batch_size):
    #         video_id = video_ids[i]
    #         frame_weights[video_id] = frame_weight[i]
    #         region_weights[video_id] = region_weight[i]


    # weights = {'frame': frame_weights, 'region': region_weights}
    # torch.save(weights, 'generated_weight/weight.pth')

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
    print('vocab size:', opt["vocab_size"])
    mode = 'beam' if opt["beam"] else 'inference'

    model = S2VT(opt)
    model = model.cuda()
    model.eval()

    checkpoint_path = os.path.join(opt["save_path"], 'checkpoint')
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
    
    for checkpoint in tqdm.tqdm(checkpoints, desc='epochs', leave=True):
        model.load_state_dict(state_dict=
            torch.load(checkpoint))

        with torch.no_grad():
            test_score = test(model, testloader, vocab, test_scorer, mode=mode)
    
        test_score.update({'epoch': checkpoint.split('/')[-1][6:9]})
        metrics_test = metrics_test.append(test_score, ignore_index=True)
        metrics_test.to_csv(metrics_log_test, float_format='%.1f', index=False)

    if len(checkpoints)==1:
        for k, v in test_score.items():
            if k != 'epoch':
                print('{}: {:.1f}'.format(k, v))


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    # set up gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]

    main(opt)