import os
import argparse
from pathlib import Path
import datetime
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--finetune_scannet',
        action='store_true',
        default=False,
        help='finetune_scannet')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)   
    parser.add_argument('--debug', action='store_true',  default=False)
    # seed
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--transfer_to_scannet', action='store_true', default=False)
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--extra_ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=10, help = 'val freq (epoch)')
    parser.add_argument('--start_val_epoch', type = int, default=100, help = 'start val epoch')
    parser.add_argument('--train_vis_per_epoch', type = int, default=5, help = 'sampling interval for visualize')
    parser.add_argument('--val_interval', type = int, default=50, help = 'sampling interval for visualize')
    parser.add_argument('--test_interval', type = int, default=50, help = 'sampling interval for visualize')
    parser.add_argument('--align', action='store_true',  default=False,  help = 'train feature aligning')
    parser.add_argument('--contrastive', action='store_true',  default=False,  help = 'contrastive learning')
    parser.add_argument('--vis_mvp', action='store_true',  default=False,  help = 'visualize selected mvps')
    parser.add_argument('--vis_list_pth', type = str, default='None',  help = 'list of visualized instances')
    # parser.add_argument('--vis', action='store_true',  default=False,  help = 'visualize val')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--finetune',
        action='store_true',
        default=False,
        help = 'finetune from start_ckpts')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')
    parser.add_argument(
        '--vis',
        action='store_true',
        default=False,
        help = 'whether to visualize results')
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.finetune and args.start_ckpts is None:
        raise ValueError(
            '--start_ckpts must be provided in finetune')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    time = datetime.datetime.now().isoformat()[:19]
    if args.test:
        args.experiment_path =  ('/').join(args.ckpts.split('/')[:-1]) + '/test/' + args.exp_name + '_' + time
    elif args.resume:
        args.experiment_path = ('/').join(args.ckpts.split('/')[:-1]) #+ '/resume/' + args.exp_name
    else:
        if args.mode is not None:
            args.exp_name = args.exp_name + '_' + args.mode
        args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,
                                            args.exp_name + '_' + time)
    #     args.exp_name = 'test_' + args.exp_name
    #args.exp_name += time.strftime('%m%d_%H%M%S', time.localtime())
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name + '_' + time)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' % args.experiment_path)
    # if not os.path.exists(args.tfboard_path):
    #     os.makedirs(args.tfboard_path, exist_ok=True)
    #     print('Create TFBoard path successfully at %s' % args.tfboard_path)

