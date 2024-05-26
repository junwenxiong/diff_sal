import torch
from mmcv import Config
from mmcv.utils.registry import build_from_cfg
from util.registry import DATASET_REGISTRY
from datasets.saliency_db import saliency_db_spec


def get_training_loader(opt):
    config_file = Config.fromfile(opt.config_file) 
    dataset = build_from_cfg(config_file["data"]["train"], DATASET_REGISTRY) 
    train_sampler = None 
    if opt.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=(train_sampler is None),
                                        num_workers=opt.n_threads,
                                        drop_last=True,
                                        pin_memory=True,
                                        sampler=train_sampler,
                                        persistent_workers=opt.n_threads>0)
    print(f"dist: {opt.multiprocessing_distributed}, using {config_file.data_type} training dataset!")
    return train_loader


def get_val_loader(opt):
    config_file = Config.fromfile(opt.config_file) 
    dataset = build_from_cfg(config_file["data"]["val"], DATASET_REGISTRY) 
    val_sampler = None
    if opt.multiprocessing_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    val_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=False,
                                        num_workers=opt.n_threads,
                                        drop_last=True,
                                        pin_memory=False,
                                        sampler=val_sampler,
                                        persistent_workers=opt.n_threads>0)
    print(f"using {config_file.data_type} val dataset!")
    return val_loader

def get_av_testset(data_config, is_training=True): 
    cur_index = data_config["index"] 
    cur_data = data_config["dataset"] 
    assert data_config["dataset"] in ['diem', 'coutrot1', 'coutrot2', 'summe', 'etmd', 'avad'] 
    flag = "test"
    print('Creating {} dataset: {}/{}'.format(flag, cur_index,  data_config["dataset"]))

    dataset = saliency_db_spec(
		data_config,
		data_config["video_path_{}".format(cur_data)],
		data_config[cur_index]["annotation_path_{}_{}".format(cur_data, flag)],
		data_config["salmap_path_{}".format(cur_data)],
		data_config["audio_path_{}".format(cur_data)],
		with_audio=data_config['with_audio'],
		use_spectrum=data_config["use_spectrum"],
		audio_type=data_config['audio_type'],
		exhaustive_sampling=True,
  		sample_duration=data_config["sample_duration"]) 

    return dataset

def get_av_dataset(data_config, is_training=True): 
    cur_index = data_config["index"] 
    cur_data = data_config["dataset"] 
    assert data_config["dataset"] in ['diem', 'coutrot1', 'coutrot2', 'summe', 'etmd', 'avad'] 
    flag = "train" if is_training else "test" 
    print('Creating {} dataset: {}/{}'.format(flag, cur_index,  data_config["dataset"]))

    dataset = saliency_db_spec(
		data_config,
		data_config["video_path_{}".format(cur_data)],
		data_config[cur_index]["annotation_path_{}_{}".format(cur_data, flag)],
		data_config["salmap_path_{}".format(cur_data)],
		data_config["audio_path_{}".format(cur_data)],
		with_audio=data_config['with_audio'],
		use_spectrum=data_config["use_spectrum"],
		audio_type=data_config['audio_type'],
		exhaustive_sampling=False if not is_training else False,
  		sample_duration=data_config["sample_duration"]) 

    return dataset

dataset_names = ['diem', 'coutrot1', 'coutrot2', 'summe', 'etmd', 'avad']
def get_test_av_loader(opt, data_config):
    dataset_list = []
    for name in dataset_names:
        data_config["dataset"] = name
        dataset_list.append(get_av_testset(data_config, is_training=False))
    dataset = torch.utils.data.ConcatDataset(dataset_list)

    if opt.multiprocessing_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    val_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=False,
                                        num_workers=opt.n_threads,
                                        drop_last=True,
                                        pin_memory=False,
                                        sampler=val_sampler,
                                        persistent_workers=opt.n_threads>0)
    print(f"dist: {opt.multiprocessing_distributed}, using saliency audio-visual test dataset!")
    return val_loader


def get_val_av_loader(opt, data_config):
    dataset_list = []
    for name in dataset_names:
        data_config["dataset"] = name
        dataset_list.append(get_av_dataset(data_config, is_training=False))
    dataset = torch.utils.data.ConcatDataset(dataset_list)

    if opt.multiprocessing_distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    val_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=False,
                                        num_workers=opt.n_threads,
                                        drop_last=True,
                                        pin_memory=False,
                                        sampler=val_sampler,
                                        persistent_workers=opt.n_threads>0)
    print(f"dist: {opt.multiprocessing_distributed}, using saliency audio-visual val dataset!")
    return val_loader

def get_training_av_loader(opt, data_config):
    dataset_list = []
    for name in dataset_names:
        data_config["dataset"] = name
        dataset_list.append(
            get_av_dataset(data_config))

    dataset = torch.utils.data.ConcatDataset(dataset_list)
    train_sampler = None 
    if opt.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               drop_last=True,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               persistent_workers=opt.n_threads>0)

    print(f"dist: {opt.multiprocessing_distributed}, using saliency audio-visual training dataset!")
    return train_loader
