import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from datasets.meta_data import MetaDataset
from util.registry import DATASET_REGISTRY


DATASET_REGISTRY.register_module()
class HollyDataset(MetaDataset):

    def __init__(self,
                 path_data,
                 len_snippet,
                 mode="train",
                 img_size=(224, 384),
                 gt_length=1,
                 alternate=1
                 ):
                 
        super(HollyDataset, self).__init__(path_data, len_snippet, mode, img_size, alternate, gt_length)

        self.vid_list = self.get_video_list(mode)

        if self.mode == "train":
            self.path_data = os.path.join(self.path_data, "training")
            self.list_num_frame = []
            self.list_video_frames = {}
            for v in self.vid_list:
                tmp_video_len = len(os.listdir(os.path.join(self.path_data, v, "images")))
                for i in range(0, tmp_video_len - self.alternate * self.len_snippet, self.skip_window):
                    self.list_num_frame.append((v, i))
                self.list_video_frames[v] = tmp_video_len
        else:
            self.path_data = os.path.join(self.path_data, "testing")
            self.list_num_frame = []
            self.list_video_frames = {}
            for v in self.vid_list:
                tmp_video_len = len(os.listdir(os.path.join(self.path_data, v, "images")))
                if tmp_video_len < self.alternate * self.len_snippet:
                    continue
                for i in range(0, tmp_video_len - self.alternate * self.len_snippet, gt_length):
                    self.list_num_frame.append((v, i))
                self.list_num_frame.append((v, tmp_video_len - self.len_snippet))
                self.list_video_frames[v] = tmp_video_len

        print(f"Holly {mode} dataset loaded! {len(self.list_num_frame)} data items!")

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        (file_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, file_name, "images")
        path_annt = os.path.join(self.path_data, file_name, "maps")

        data = {'rgb': []}
        target = {'salmap': []}

        # TODO 继续测试在16帧参数的数据集上的效果
        if self.len_snippet > 16:
            frame_lens = 16
            indices = [start_idx + self.alternate * i for i in range(frame_lens)]
        else:
            indices = [start_idx + self.alternate * i for i in range(self.len_snippet) ]
            
        clip_img = []
        img_list = sorted(os.listdir(path_clip))
        for i in indices:
            img_name = img_list[i]
            img = Image.open(os.path.join(path_clip, img_name)).convert('RGB')
            clip_img.append(self.img_transform(img))
        clip = torch.stack(clip_img, 0).permute(1, 0, 2, 3)
        clip_img = torch.FloatTensor(clip)

        def get_center_slice(arr, length):
            center = len(arr) // 2  # 获取数组的中心位置索引
            start = center - length // 2  # 计算起始索引
            end = start + length  # 计算结束索引
            return arr[start:end]

        clip_gt = None
        if self.mode != "save" and self.mode != "test":
            gt_sequence_list = get_center_slice(indices, self.gt_length)
            gt_maps_list = []
            for gt_index in gt_sequence_list:
                mid_img_name = img_list[gt_index]
                gt_index_path = os.path.join(path_annt, mid_img_name)
                gt_maps_list.append(self.load_gt_PIL(gt_index_path))
            gt_maps = torch.stack(gt_maps_list, 0).permute(1, 0, 2, 3).squeeze(1)
            clip_gt = gt_maps

        data['rgb'] = clip_img
        data["video_id"] = file_name

        data["video_index"] = file_name     # 用于预测
        data["gt_index"] =  torch.tensor(gt_sequence_list)
        target['salmap'] = clip_gt

        return data, target

if __name__ == '__main__':
    
    train_data = HollyDataset(path_data="VideoSalPrediction/Hollywood2",
                     len_snippet=16, 
                     mode="test")

    tmp_data = train_data.__getitem__(0)
    print(train_data.__len__())

    from torch.utils.data import DataLoader

    data_loader = DataLoader(train_data, batch_size=16, num_workers=32)
    for batch, target in data_loader:
        # print(batch["rgb"].shape)
        # print(target["salmap"].shape)
        print(batch["video_id"], batch["gt_index"])