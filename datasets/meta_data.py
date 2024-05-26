import torch
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms

class MetaDataset(data.Dataset):
    def __init__(self,
                 path_data,
                 len_snippet,
                 mode="train",
                 img_size = (224, 224),
                 alternate=1,
                 gt_length=1):
        super(MetaDataset, self).__init__()

        ''' mode: train, val, test '''
        self.path_data = path_data
        self.len_snippet = len_snippet
        self.mode = mode
        self.gt_length = gt_length
        self.alternate = alternate
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.sal_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        # TODO skip_window的值太大了，可以减小点
        if self.len_snippet > 16:
            self.skip_window = self.len_snippet // 2
            self.skip_window = 16
        else:
            self.skip_window = self.len_snippet
    
    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return len(self.list_num_frame)
     
    def load_gt_PIL(self, path):
        gt = Image.open(path).convert('L')
        gt = self.sal_transform(gt)
        return gt

    def load_gt(self, path):
        gt = np.array(Image.open(path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (self.img_size[-1], self.img_size[0]))
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        return gt

    def get_groundtruth(self, path1, path2): 
        gt = self.load_gt(path1) 
        mid_gt =torch.FloatTensor(gt).unsqueeze(0) 
        gt4 = self.load_gt(path2) 
        futr_gt = torch.FloatTensor(gt4).unsqueeze(0) 

        return mid_gt, futr_gt

    def get_multi_gt_maps2(self, gt_index_list, gt_path):
        gt_maps_list = []
        for gt_index in gt_index_list:
            gt_index_path = os.path.join(gt_path, '%04d.png' % int(gt_index))
            gt_maps_list.append(self.load_gt_PIL(gt_index_path))

        clip = torch.stack(gt_maps_list, 0).permute(1, 0, 2, 3).squeeze(1)
        return torch.FloatTensor(clip)
        # return torch.from_numpy(np.asarray(gt_maps_list)/255.0)
    
    def get_multi_gt_maps(self, gt_index_list, gt_path):
        gt_maps_list = []
        for gt_index in gt_index_list:
            gt_index_path = os.path.join(gt_path, '%04d.png' % int(gt_index))
            gt_maps_list.append(self.load_gt(gt_index_path))

        return torch.from_numpy(np.asarray(gt_maps_list)/255.0)

    def get_video_list(self, mode):
        if mode == 'train': 
            path = '{}/{}'.format(self.path_data, 'training')
        else: 
            path = '{}/{}'.format(self.path_data, 'testing')

        video_list = os.listdir(path)
        video_list = [v for v in video_list]

        return video_list