import numpy as np
import csv
import math
import subprocess
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim



def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = np.clip((data - data_min) * (255.0 / (data_max - data_min)), 0,
                        255).astype(np.uint8)
    return data_norm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterList(object):
    """Computes and stores the average and current value"""

    def __init__(self, name_list=['main', 'cc', 'sim', 'nss', 'total']):
        metric_dict = {}
        for name in name_list:
            metric_dict[name] = AverageMeter()
        
        self.metrics = metric_dict

    def update(self, loss_dict):
        assert loss_dict.keys() == self.metrics.keys()
        for k in self.metrics.keys():
            self.metrics[k].update(round(loss_dict[k].item(), 3))
        
    def get_metric(self, key: str=None):
        return self.metrics[key]


class LogWritter:
    def __init__(self, path):
        self.path = path

    def update_txt(self, msg, mode='a+'):
        if mode == 'w':
            fp = open('{}'.format(self.path), mode)
            print(msg, file=fp)
            fp.close()
        elif mode == 'a+':
            print(msg)
            fp = open('{}'.format(self.path), mode)
            print(msg, file=fp)
            fp.close()
        else:
            raise Exception('other file operation is unimplemented !')

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a+')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            if col not in values:
                write_values.append(0.0)
            else:
                write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def get_optim_scheduler(config, parameters, num_epoches):
    optimizer = optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epoches * 0.5),
                                                                int(0.75 * num_epoches)], 
                                         gamma=0.1, verbose=True)
    return optimizer, scheduler

    
def compress_files(root_path):
    zip_path = root_path.split("/")[-2]
    zip_name = root_path.split("/")[-1]
    print(zip_path, zip_name)

    cmd_metrics = r"""
    zip_path={}
    zip_name={}
    cd $zip_path
    zip -r $zip_name.zip $zip_name/split1_results/ $zip_name/split2_results/ $zip_name/split3_results/
    """.format(zip_path, zip_name)

    status = subprocess.run(cmd_metrics, shell=True, check=True)
