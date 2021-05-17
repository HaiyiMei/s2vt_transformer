import argparse

dim = {
    'resnext': 2048,
    'resnet': 2048,
    'tsn': 2048,
    'inception': 1536,
    'incep_res': 1536,
    'i3d': 2048,
    'senet': 2048,
    'pnasnet': 4320,
    'vgg': 512
}

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
        '--save_path',
        type=str,
        default='save',
        help='directory to store checkpointed models')
    parser.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=None,
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
        "--model_box", type=str, default='', help="which box model to use")
    parser.add_argument(
        '--decoder',
        type=str,
        default='transformer',
        help='whether to use transformer/lstm as decoder')
    parser.add_argument(
        '--transformer_encoder',
        action='store_true',
        help='whether to use transformer as encoder')
    parser.add_argument(
        '--only_box',
        action='store_true',
        help='whether to use box features')
    parser.add_argument(
        '--wcm',
        type=str,
        default='channel',
        help='which way to use wcm [channel/scalar/fc]')
    parser.add_argument(
        '--wcm_len',
        type=int,
        default=1,
        help='how many wcm [channel/scalar/fc] to use')
    parser.add_argument(
        '--fusion',
        type=str,
        help='which guide [concat/add] mode to use')
    parser.add_argument(
        '--cbp_dim',
        type=int,
        default=4000)
    parser.add_argument(
        '--sample_len',
        type=int,
        default=32)
    parser.add_argument(
        '--n_layer',
        type=int,
        default=1,
        help='layer number for transformer layer')
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
        '--epochs', type=int, default=1001, help='number of epochs')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5,  # 5.,
        help='clip gradients at this value')
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=3e-4, 
        help='learning rate')
    parser.add_argument(
        '-j', '--workers', 
        default=4, 
        type=int, 
        metavar='N',
        help='number of data loading workers (default: 4)')
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

    # t = time.strftime("%m-%d-%H-%M")

    args.save_path = os.path.join(args.save_path, *args.feats_dir.split('/')[-2:])
    args.save_path = os.path.join(args.save_path, args.model)
    if args.decoder == 'lstm':
        args.save_path = args.save_path + '_lstm'
    if args.only_box:
        args.save_path = os.path.join(args.save_path, args.model_box)
        args.save_path = args.save_path + '_onlyBox'
    if args.fusion:
        args.save_path = f'{args.save_path}_{args.model_box}BOX'
    
    args.save_path = os.path.join(args.save_path, f'{args.wcm}_{args.wcm_len}_{args.fusion}')

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
    if args.model_box == '':
        args.model_box = args.model
    if args.save_path=='save':
        args = process_checkpoint(args)
    args.dim_vid = dim[args.model]
    args.dim_box = dim[args.model_box]

    return args

if __name__ == '__main__':
    import json
    opt = parse_opt()
    opt = vars(opt)
    print(json.dumps(opt, indent=1))
