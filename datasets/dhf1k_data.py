import torch
import os
from PIL import Image
from datasets.meta_data import MetaDataset
from util.registry import DATASET_REGISTRY


DATASET_REGISTRY.register_module()
class DHF1KDatasetMultiFrames(MetaDataset):

    def __init__(self,
                 path_data,
                 len_snippet,
                 mode="train",
                 img_size=(224, 224),
                 gt_length=1,
                 alternate=1):

        super(DHF1KDatasetMultiFrames, self).__init__(path_data, len_snippet, mode, img_size, alternate, gt_length)
        ''' mode: train, val, test '''

        self.img_path = os.path.join(self.path_data, "frames")
        self.ann_path = os.path.join(self.path_data, "maps")

        def extract_integer(filename):
            return int(filename)

        self.video_names = sorted(os.listdir(self.img_path), key=extract_integer)
        self.video_train_names = self.video_names[:int(600)]
        self.video_val_names = self.video_names[int(600):int(700)]
        self.video_test_names = self.video_names[int(700):int(1000)]

        if self.mode == "train":
            # 隔len_snippet取一次
            self.list_num_frame = []
            for v in self.video_train_names:
                tmp_video_len = len(os.listdir(os.path.join(self.img_path, v)))
                for i in range(0,  tmp_video_len - self.alternate * self.len_snippet, self.skip_window):
                    self.list_num_frame.append((v, i))
        elif self.mode == "val":
            self.list_num_frame = []
            self.list_video_frames = {}
            for v in self.video_val_names:
                tmp_video_len = len(os.listdir(os.path.join(self.img_path, v)))
                for i in range(0, tmp_video_len - self.alternate * self.len_snippet, self.gt_length):
                    self.list_num_frame.append((v, i))
                self.list_video_frames[v] = tmp_video_len
        else:
            self.list_num_frame = []
            self.list_video_frames = {}
            for v in self.video_test_names:
                tmp_video_len = len(os.listdir(os.path.join(self.img_path, v)))
                for i in range(0, tmp_video_len - self.alternate * self.len_snippet, 1):
                    self.list_num_frame.append((v, i))
                self.list_num_frame.append((v, tmp_video_len - self.len_snippet))
                self.list_video_frames[v] = tmp_video_len

        print(f"DHF1k {mode} dataset loaded! {len(self.list_num_frame)} data items!")

    def __getitem__(self, idx):
        (file_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.img_path, file_name)
        path_annt = os.path.join(self.ann_path, file_name)

        data = {'rgb': []}
        target = {'salmap': []}

        # TODO 继续测试在16帧参数的数据集上的效果
        if self.len_snippet > 16:
            frame_lens = 16
            indices = [start_idx + self.alternate * i + 1 for i in range(frame_lens)]
        else:
            indices = [start_idx + self.alternate * i + 1 for i in range(self.len_snippet) ]
            
        clip_img = []
        for i in indices:
            img = Image.open(os.path.join(path_clip, '%d.png' % i)).convert('RGB')
            clip_img.append(self.img_transform(img))
        clip = torch.stack(clip_img, 0).permute(1, 0, 2, 3)
        clip_img = torch.FloatTensor(clip)
        clip_gt = None

        def get_center_slice(arr, length):
            center = len(arr) // 2  # 获取数组的中心位置索引
            start = center - length // 2  # 计算起始索引
            end = start + length  # 计算结束索引
            return arr[start:end]

        # ground_truth
        if self.mode != "save" and self.mode != "test":
            gt_sequence_list = get_center_slice(indices, self.gt_length)
            gt_maps = self.get_multi_gt_maps2(gt_sequence_list, path_annt) # (len(seq), 224, 384)
            clip_gt = gt_maps

        data['rgb'] = clip_img
        data["video_id"] = torch.tensor(int(file_name)) # 用于训练
        data["video_index"] = file_name     # 用于预测
        data["gt_index"] =  torch.tensor(gt_sequence_list)

        if self.mode == 'val':
            target['salmap'] = clip_gt
        elif self.mode == "test":
            target['salmap'] = 0
        else:
            target['salmap'] = clip_gt

        return data, target


if __name__ == "__main__":

    new_dataset = DHF1KDatasetMultiFrames(path_data="VideoSalPrediction/DHF1k_extracted",
                                 len_snippet=48,
                                 mode="val")

    tmp_data = new_dataset.__getitem__(0)
    print(new_dataset.__len__())

    from torch.utils.data import DataLoader

    # import pdb; pdb.set_trace()
    data_loader = DataLoader(new_dataset, batch_size=16, num_workers=8)
    for batch, target in data_loader:
        # print(batch["rgb"].shape)
        # print(target["salmap"].shape)
        # print(batch["video_id"], batch["gt_index"])
        print(target["salmap"]["futr"].shape)
        print(batch["gt_index"])
