import argparse


def add_basic_options(parser):
    parser.add_argument(
        '--feats_dir',
        type=str,
        default='feats/MSVD/uniform_batch',
        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument(
        '--save_checkpoint',
        action='store_true',
        help='whether to save checkpoint')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='save',
        help='directory to store checkpointed models')
    parser.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=50,
        help='directory to store checkpointed models')
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument('--cfg', type=str, default=None,
            help='configuration; similar to what is used in detectron')
    return parser


def add_model_options(parser):
    # Model settings
    parser.add_argument(
        "--model", type=str, default='tsn', help="which model to use")
    parser.add_argument(
        '--transformer',
        action='store_true',
        help='whether to use transformer as decoder')
    parser.add_argument(
        '--transformer_encoder',
        action='store_true',
        help='whether to use transformer as encoder')
    parser.add_argument(
        '--attention',
        action='store_true',
        help='whether to use attnetion in decoder')
    parser.add_argument(
        '--with_box',
        action='store_true',
        help='whether to use box features')
    parser.add_argument(
        '--only_box',
        action='store_true',
        help='whether to use box features')
    parser.add_argument(
        '--tg',
        action='store_true',
        help='whether to use time gcn')
    parser.add_argument(
        '--bg',
        action='store_true',
        help='whether to use box gcn')
    # parser.add_argument(
    #     '--guide',
    #     action='store_true',
    #     help='whether to use guide gcn')
    parser.add_argument(
        '--fusion',
        type=str,
        help='which guide [channel, concat/add] mode to use')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=1,
        help='layer number for transformer layer')
    parser.add_argument(
        '--n_layer_fusion',
        type=int,
        default=1,
        help='layer number for transformer decoder')
    return parser

def add_training_options(parser):
    parser.add_argument(
        '--self_crit_after',
        type=int,
        default=-1,
        help='After what epoch do we start finetuning the CNN? \
             (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument(
        "--max_len",
        type=int,
        default=28,
        help='max length of captions(containing <sos>,<eos>)')
    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=1024,
        help='size of the rnn hidden layer')
    parser.add_argument(
        '--dim_vid',
        type=int,
        default=2048,
        help='dim of features of video frames')
    parser.add_argument(
        '--epochs', type=int, default=1001, help='number of epochs')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5,  # 5.,
        help='clip gradients at this value')
    parser.add_argument(
        '--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='select random seed')
    parser.add_argument(
        '--beam',
        action='store_true',
        help='whether to use beam search')
    parser.add_argument(
        '--warmup',
        type=int,
        default=-1,
        help='use warmup learning strategy (in epoch, -1 disable)')
    parser.add_argument(
        '--eval_start',
        type=int,
        default=500,
        help='when to start evaluation (in epoch)?')
    return parser


def process_checkpoint(args):
    import time
    import os

    t = time.strftime("%H:%M:%S")

    args.checkpoint_path = os.path.join(args.checkpoint_path, *args.feats_dir.split('/')[-2:])
    # args.checkpoint_path = os.path.join(args.checkpoint_path, os.path.basename(args.feats_dir))
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.model)
    if args.transformer:
        args.checkpoint_path = args.checkpoint_path + '_TRAN'
    if args.tg:
        args.checkpoint_path = args.checkpoint_path + '_tg'
    if args.with_box:
        args.checkpoint_path = args.checkpoint_path + '_box'
    if args.only_box:
        args.checkpoint_path = args.checkpoint_path + '_OnlyBox'
    if args.bg:
        args.checkpoint_path = args.checkpoint_path + '(bg)'
    if args.attention:
        args.checkpoint_path = args.checkpoint_path + '_ATT'
    if args.fusion:
        args.checkpoint_path = args.checkpoint_path + '_fusion'
        t = t + '_' + str(args.fusion)
    
    args.checkpoint_path = os.path.join(args.checkpoint_path, t)

    return args

def parse_opt():
    parser = argparse.ArgumentParser()
    parser = add_basic_options(parser)
    parser = add_model_options(parser)
    parser = add_training_options(parser)

    # read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None:
        from misc.config import CfgNode
        cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' %k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    args.dataset = 'MSRVTT' if 'MSRVTT' in args.feats_dir else 'MSVD'
    if args.checkpoint_path=='save':
        args = process_checkpoint(args)

    return args

if __name__ == '__main__':
    import json
    opt = parse_opt()
    opt = vars(opt)
    print(json.dumps(opt, indent=1))