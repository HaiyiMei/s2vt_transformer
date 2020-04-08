import numpy as np
from collections import OrderedDict
import torch
import sys
sys.path.append("coco-caption")
from pyciderevalcap.ciderD.ciderD import CiderD
# from pyciderevalcap.cider.cider import Cider

CiderD_scorer = None
# CiderD_scorer = CiderD(df='corpus')


def init_cider_scorer(dataset):
    cached_tokens = '{}-idxs'.format(dataset)
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] == 0:
            break
        if arr[i] == 1:
            continue
        out += str(arr[i]) + ' '
    return out.strip()


def get_self_critical_reward(greedy_res, data, gen_result):
    gen_result = gen_result.cpu().data.numpy()
    greedy_res = greedy_res.cpu().data.numpy()
    data_gts = data['gts'].cpu().data.numpy()

    batch_size = len(data_gts) 
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})

    _, scores = CiderD_scorer.compute_score(gts_, res_)
    # print('Cider scores:', _)

    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

# def get_self_critical_reward(greedy_res, data, gen_result):
#     batch_size = gen_result.size(0)

#     gen_result = gen_result.cpu().data.numpy()
#     greedy_res = greedy_res.cpu().data.numpy()
#     data_gts = data['gts'].cpu().data.numpy()

#     res = OrderedDict()
#     for i in range(batch_size):
#         res[i] = [array_to_str(gen_result[i])]
#     for i in range(batch_size):
#         res[batch_size + i] = [array_to_str(greedy_res[i])]

#     gts = OrderedDict()
#     for i in range(len(data_gts)):
#         gts[i] = [array_to_str(data_gts[i][j])
#                   for j in range(data_gts.shape[1])]

#     res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
#     gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
#     # _, scores = CiderD_scorer.compute_score(gts, res)
#     _, scores = CiderD_scorer.compute_score(gts, res)
#     # print('Cider scores:', _)

#     scores = scores[:batch_size] - scores[batch_size:]
#     # scores = scores[:batch_size]
#     # print(scores.shape)
#     # print(gen_result.shape)

#     rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

#     return rewards
