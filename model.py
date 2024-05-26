import torch
from torch import nn
from util.registry import OBJECT_REGISTRY
from mmcv.utils.registry import build_from_cfg
from mmcv import Config

def generate_av_model(opt):
    cfg = Config.fromfile(opt.config_file)
    model = build_from_cfg(cfg['config'], OBJECT_REGISTRY)
    if not opt.multiprocessing_distributed:
        model = nn.DataParallel(model.cuda())
    else:
        torch.cuda.set_device(opt.gpu)
        model.cuda(opt.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)

    if opt.pretrain_path.strip() != "":
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(pretrain["state_dict"], strict=0)
        print(msg)
        del pretrain

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    return model, parameters