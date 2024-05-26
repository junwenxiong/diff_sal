import argparse
import yaml
import os

def parse_opts():
    parser = argparse.ArgumentParser(
        description='options and parameters')

    # Root and results path
    parser.add_argument('--dist',
                        action='store_true',
                        help='If True, use the distribution training.')
    parser.add_argument('--root_path',
                        default='./experiments/',
                        type=str,
                        help='Root directory path of experiments')
    parser.add_argument('--config_file',
                        default='cfgs/visual.py',
                        type=str,
                        help='Root directory path of experiments')
    parser.add_argument('--gpu_devices',
                        default='0,1,2,3',
                        type=str,
                        help='GPU devices ids')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='If True, use the wandb')
    parser.add_argument('--data_type',
                        choices=["dhf1k", "holly", "ucf", "av_data"],
                        default='dhf1k',
                        help='If True, use the dhf1k.')
    parser.add_argument('--lr_scheduler',
                        default="MultiStepLR",
                        choices=["MultiStepLR", "CosineAnnealingLR"],
                        help='If True, use the aspp.')
    parser.add_argument(
        '--name',
        default='0001',
        type=str,
        help='experiment name')

    parser.add_argument('--pretrain_path',
                        default='',
                        type=str,
                        help='Pretrained visual model (.pth)')

    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch size dependent on GPUs memory')

    parser.add_argument('--train',
                        action='store_true',
                        help='If true, training is performed.')
    parser.add_argument('--test',
                        action='store_true',
                        help='If true, test is performed.')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')

    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    # Distributed params
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url',
                        default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    # diffusion
    parser.add_argument("--resume_training",
                        action="store_true",
                        help="Whether to resume training")
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to the config file")

    args = parser.parse_args()

    os.makedirs(args.root_path, exist_ok=True)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    new_config = dict2namespace(config)

    with open(os.path.join(args.root_path, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace