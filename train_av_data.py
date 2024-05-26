import os
import shutil
import torch
from util.opts import parse_opts
import wandb
from torch import multiprocessing as mp
from diffusion_trainer import DiffusionTrainer
import warnings

warnings.filterwarnings("ignore")


def wandb_init(opt):
    wandb.init(
        config=opt,
        group="xxx",
        project="xxx",
        name=opt.name,
    )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(opt):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        opt.rank = int(os.environ["RANK"])
        opt.world_size = int(os.environ["WORLD_SIZE"])
        opt.gpu = int(os.environ["LOCAL_RANK"])

    opt.distributed = True
    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, word {}): {}".format(
            opt.rank, opt.world_size, opt.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=opt.dist_backend,
        init_method=opt.dist_url,
        world_size=opt.world_size,
        rank=opt.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(opt.rank == 0)
    print("finished")


def main_worker(gpu, opt, config=None):
    opt.gpu = gpu
    init_distributed_mode(opt)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    shutil.copy(opt.config_file, opt.root_path)
    torch.manual_seed(opt.manual_seed)

    wad = None
    if opt.wandb:
        wad = wandb_init(opt)
    result_path = "results"
    weight_path = "weights"
    opt.result_path = os.path.join(opt.root_path, result_path)
    opt.weight_path = os.path.join(opt.root_path, weight_path)

    os.makedirs(opt.result_path, exist_ok=True)
    os.makedirs(opt.weight_path, exist_ok=True)

    runner = DiffusionTrainer(opt, config)
    if opt.train:
        runner.train_av_data()
    if opt.test:
        runner.test_av_data(save_img=True)


def main(opt, config=None):
    mp.set_start_method("spawn")
    main_worker(opt.gpu, opt, config)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False  # type: bool
    opt, config = parse_opts()

    main(opt, config)
