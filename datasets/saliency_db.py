import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import os, json
import functools
import copy
import numpy as np
from numpy import median
import scipy.io as sio
from scipy import signal
import torchaudio
import soundfile as sf
from torchvision import transforms
import torchaudio.functional as F
from datasets.torchvggish.vggish_input import waveform_to_examples
from decimal import localcontext, Decimal, ROUND_HALF_UP
from datasets.spatial_transforms import (Compose, Normalize, Scale,
                                         RandomHorizontalFlip, ToTensor)
from datasets.temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from datasets.target_transforms import Label, VideoID

ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return  samples

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            new_size = (320, 240)
            img = img.resize(new_size)
            return img


def pil_loader_sal(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('L')
            return img


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def read_sal_text(txt_file):
    test_list = {'names': [], 'nframes': [], 'fps': []}
    with open(txt_file, 'r') as f:
        for line in f:
            word = line.split()
            test_list['names'].append(word[0])
            test_list['nframes'].append(word[1])
            test_list['fps'].append(word[2])
    return test_list

def make_dataset(root_path, annotation_path, salmap_path, audio_path, step,
                 step_duration):
    data = read_sal_text(annotation_path)
    video_names = data['names']
    video_nframes = data['nframes']
    video_fps = data['fps']
    dataset = []
    audiodata = []
    for i in range(len(video_names)):
        video_path = os.path.join(root_path, video_names[i])
        annot_path = os.path.join(salmap_path, video_names[i], 'maps')
        annot_path_bin = os.path.join(salmap_path, video_names[i])
        if not os.path.exists(video_path):
            continue
        if not os.path.exists(annot_path):
            continue
        if not os.path.exists(annot_path_bin):
            continue

        n_frames = int(video_nframes[i])
        if n_frames <= 1:
            continue

        begin_t = 1
        end_t = n_frames

        audio_wav_path = os.path.join(audio_path, video_names[i],
                                      video_names[i] + '.wav')
        if not os.path.exists(audio_wav_path):
            continue

        target_Fs = 16000
        [audiowav, Fs] = torchaudio.load(audio_wav_path)
        audiowav = F.resample(audiowav, Fs, target_Fs)
        Fs = target_Fs

        n_samples = Fs / float(video_fps[i])
        starts = np.zeros(n_frames + 1, dtype=int)
        ends = np.zeros(n_frames + 1, dtype=int)
        starts[0] = 0
        ends[0] = 0

        for videoframe in range(1, n_frames + 1):
            startemp = max(0,
                           ((videoframe - 1) *
                            (1.0 / float(video_fps[i])) * Fs) - n_samples / 2)
            starts[videoframe] = int(startemp)
            endtemp = min(
                audiowav.shape[1],
                abs(((videoframe - 1) * (1.0 / float(video_fps[i])) * Fs) +
                    n_samples / 2))
            ends[videoframe] = int(endtemp)

        audioinfo = {
            'audiopath': audio_path,
            'video_id': video_names[i],
            'Fs': Fs,
            'wav': audiowav,
            'starts': starts,
            'ends': ends
        }
        audiodata.append(audioinfo)

        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'fps': video_fps[i],
            'video_id': video_names[i],
            'salmap': annot_path,
            'binmap': annot_path_bin
        }
        step = int(step)
        for j in range(1, n_frames, step):
            sample_j = copy.deepcopy(sample)
            sample_j['frame_indices'] = list(
                range(j, min(n_frames + 1, j + step_duration)))
            dataset.append(sample_j)

    print('dataset loading [{}/{}] successfully!'.format(i+1, len(video_names)))

    return dataset, audiodata

def make_mel_dataset(root_path, annotation_path, salmap_path, audio_path, step,
                     step_duration):
    data = read_sal_text(annotation_path)
    video_names = data['names']
    video_nframes = data['nframes']
    video_fps = data['fps']
    dataset = []
    audiodata = []
    for i in range(len(video_names)):
        video_path = os.path.join(root_path, video_names[i])
        annot_path = os.path.join(salmap_path, video_names[i], 'maps')
        annot_path_bin = os.path.join(salmap_path, video_names[i])
        if not os.path.exists(video_path):
            continue
        if not os.path.exists(annot_path):
            continue
        if not os.path.exists(annot_path_bin):
            continue

        n_frames = int(video_nframes[i])
        if n_frames <= 1:
            continue

        begin_t = 1
        end_t = n_frames

        audio_wav_path = os.path.join(audio_path, video_names[i],
                                      video_names[i] + '.wav')
        if not os.path.exists(audio_wav_path):
            continue

        audiowav, Fs = sf.read(audio_wav_path, dtype='int16')
        assert audiowav.dtype == np.int16, 'Bad sample type: %r' % audiowav.dtype
        audiowav = audiowav / 32768.0

        n_samples = Fs / float(video_fps[i])
        starts = np.zeros(n_frames + 1, dtype=int)
        ends = np.zeros(n_frames + 1, dtype=int)
        starts[0] = 0
        ends[0] = 0
        for videoframe in range(1, n_frames + 1):
            startemp = max(0,
                           ((videoframe - 1) *
                            (1.0 / float(video_fps[i])) * Fs) - n_samples / 2)
            starts[videoframe] = int(startemp)
            endtemp = min(
                audiowav.shape[0],
                abs(((videoframe - 1) * (1.0 / float(video_fps[i])) * Fs) +
                    n_samples / 2))
            ends[videoframe] = int(endtemp)

        audioinfo = {
            'audiopath': audio_path,
            'video_id': video_names[i],
            'Fs': Fs,
            'wav': audiowav,
            'starts': starts,
            'ends': ends
        }
        audiodata.append(audioinfo)

        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'fps': video_fps[i],
            'video_id': video_names[i],
            'salmap': annot_path,
            'binmap': annot_path_bin
        }
        step = int(step)
        for j in range(1, n_frames, step):
            sample_j = copy.deepcopy(sample)
            sample_j['frame_indices'] = list(
                range(j, min(n_frames + 1, j + step_duration)))
            dataset.append(sample_j)

    print('dataset loading [{}/{}] successfully!'.format(i+1, len(video_names)))

    return dataset, audiodata

class saliency_db_spec(data.Dataset):
    def __init__(self,
                 data_config,
                 root_path,
                 annotation_path,
                 subset,
                 audio_path,
                 with_audio=False,
                 exhaustive_sampling=False,
                 use_spectrum=False,
                 audio_type="ori",
                 sample_duration=16,
                 step_duration=90,
                 get_loader=get_default_video_loader): 

        if exhaustive_sampling:
            self.exhaustive_sampling = True
            step = 1
            step_duration = sample_duration
        else:
            self.exhaustive_sampling = False
            step = max(1, step_duration - sample_duration)

        self.with_audio = with_audio
        self.annotation_path = annotation_path
        self.use_spectrum = use_spectrum
        self.audio_type = audio_type
        if self.audio_type == 'mel' or self.audio_type == 'spec':
            self.data, self.audiodata = make_mel_dataset(
                root_path, annotation_path, subset, audio_path, step,
                step_duration)
        else:
            self.data, self.audiodata = make_dataset(root_path,
                                                     annotation_path, subset,
                                                     audio_path, step,
                                                     step_duration)


        spatial_transform = Compose([
            Scale([data_config["sample_size"][0], data_config["sample_size"][1]]),
            ToTensor(data_config["norm_value"]), 
            Normalize(data_config["mean"], data_config["std"])
        ])
        temporal_transform = TemporalCenterCrop(data_config["sample_duration"])
        target_transform = transforms.Compose([
                transforms.Resize((data_config["sample_size"][1], data_config["sample_size"][0])),
                transforms.ToTensor(),
            ])

        audio_transform = transforms.Compose([
                transforms.Resize((data_config["sample_size"][1]//2, data_config["sample_size"][0]//2)),
        ])
        self.audio_transform = audio_transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform =  target_transform

        self.loader = get_loader()

        max_audio_Fs = 22050
        min_video_fps = 10
        self.max_audio_win = int(max_audio_Fs / min_video_fps *
                                 sample_duration)

    def __getitem__(self, index):
        path = self.data[index]['video']
        annot_path = self.data[index]['salmap']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        video_name = self.data[index]['video_id']
        flagexists = False
        audioind = None
        for iaudio in range(0, len(self.audiodata)):
            if (video_name == self.audiodata[iaudio]['video_id']):
                audioind = iaudio
                flagexists = True
                break

        if not flagexists:
            print(video_name)

        data = {'rgb': [], 'audio': []}
        # get audio 
        frame_ind_start = frame_indices[0]  # start index
        frame_ind_end = frame_indices[len(frame_indices) - 1]  # end index

        if self.with_audio:
            if self.audio_type == 'mel':
                mel_feat = self.get_mel_feature(audioind,
                                                frame_ind_start,
                                                frame_ind_end,
                                                self.max_audio_win,
                                                is_exist=flagexists)

                mel_list = []
                for mel in mel_feat:
                    mel_list.append(self.audio_transform(mel))
                data['audio'] = torch.stack(mel_list, dim=1)

            elif self.audio_type == 'spec':
                data['audio'] = self.get_spec_feature(audioind,
                                                   frame_ind_start,
                                                   frame_ind_end,
                                                   self.max_audio_win,
                                                   is_exist=flagexists)
            elif self.audio_type == 'ori':
                data['audio'] = self.get_audio_feature(audioind,
                                                    frame_ind_start,
                                                    frame_ind_end,
                                                    self.max_audio_win,
                                                    is_exist=flagexists)

        with localcontext() as ctx:
            ctx.rounding = ROUND_HALF_UP
            med_indices = Decimal(median(frame_indices)) # median index for label
            med_indices = int(med_indices.to_integral_value())

        target = {'salmap': []}
        target['salmap'] = pil_loader_sal(
            os.path.join(annot_path, 'eyeMap_{:05d}.jpg'.format(med_indices)))
        if self.exhaustive_sampling:
            dataset_name = path.split('/')[-2]
            data['video_index'] = f"{dataset_name}/{self.data[index]['video_id']}"
            data['gt_index'] = torch.tensor(med_indices)

        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            self.spatial_transform_sal = copy.deepcopy(self.spatial_transform)
            del self.spatial_transform_sal.transforms[-1]
            clip = [self.spatial_transform(img) for img in clip]

        target['salmap'] = self.target_transform(target['salmap'])
        if target['salmap'].max()==0:
            tmp_index = np.random.randint(0,index-1)
            return self.__getitem__(tmp_index)

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        data['rgb'] = clip

        return data, target

    def __len__(self):
        return len(self.data)

    def get_spec_feature(self,
                        audioind,
                        start_idx,
                        end_idx,
                        max_audio_win,
                        is_exist=True):

        audio_feature = torch.zeros(1, 257, 219)  ## maximum audio excerpt duration

        if not is_exist:
            return audio_feature

        valid = {}
        valid['audio'] = 0
        audioexcer = np.zeros((max_audio_win, ))

        excerptstart = self.audiodata[audioind]['starts'][start_idx]
        excerptend = self.audiodata[audioind]['ends'][end_idx]

        try:
            valid['audio'] = self.audiodata[audioind]['wav'][excerptstart:excerptend + 1].shape[0]
        except:
            pass

        audioexcer_tmp = self.audiodata[audioind]['wav'][excerptstart:excerptend + 1]
        samplerate = self.audiodata[audioind]['Fs']

        if (valid['audio'] % 2) == 0:
            audioexcer[((audioexcer.shape[0]//2)-(valid['audio']//2)):((audioexcer.shape[0]//2)+(valid['audio']//2))] = \
                audioexcer_tmp
        else:
            audioexcer[((audioexcer.shape[0]//2)-(valid['audio']//2)):((audioexcer.shape[0]//2)+(valid['audio']//2)+1)] = \
                audioexcer_tmp

        audioexcer[audioexcer > 1.] = 1.
        audioexcer[audioexcer < -1.] = -1.

        frequencies, times, spectrogram = signal.spectrogram(audioexcer, samplerate, nperseg=512,noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean,std+1e-9)
        spectrogram = torch.FloatTensor(spectrogram[np.newaxis, ...])

        return spectrogram

    def get_mel_feature(self,
                        audioind,
                        start_idx,
                        end_idx,
                        max_audio_win,
                        is_exist=True):

        audio_len = 9
        audio_feature = torch.zeros(audio_len, 1, 64,
                                    64)  ## maximum audio excerpt duration

        if not is_exist:
            return audio_feature

        valid = {}
        valid['audio'] = 0
        audioexcer = np.zeros((max_audio_win, ))

        excerptstart = self.audiodata[audioind]['starts'][start_idx]
        excerptend = self.audiodata[audioind]['ends'][end_idx]

        try:
            valid['audio'] = self.audiodata[audioind]['wav'][
                excerptstart:excerptend + 1].shape[0]
        except:
            pass

        audioexcer_tmp = self.audiodata[audioind]['wav'][
            excerptstart:excerptend + 1]

        if (valid['audio'] % 2) == 0:
            audioexcer[((audioexcer.shape[0]//2)-(valid['audio']//2)):((audioexcer.shape[0]//2)+(valid['audio']//2))] = \
                audioexcer_tmp
        else:
            audioexcer[((audioexcer.shape[0]//2)-(valid['audio']//2)):((audioexcer.shape[0]//2)+(valid['audio']//2)+1)] = \
                audioexcer_tmp

        sample_rate = self.audiodata[audioind]['Fs']
        audio_feature = waveform_to_examples(audioexcer, sample_rate, return_tensor=True)
        
        if audio_feature.shape[0] != audio_len:
            len_add = audio_len // audio_feature.shape[0]
            len_add2 = audio_len % audio_feature.shape[0]

            if len_add != 0:
                audio_feature = torch.repeat_interleave(audio_feature, len_add, dim=0)
            audio_feature = torch.cat([audio_feature, audio_feature[:len_add2]], dim=0)

        return audio_feature[:audio_len]

    def get_audio_feature(self,
                          audioind,
                          start_idx,
                          end_idx,
                          max_audio_win,
                          is_exist=True):

        audioexcer = torch.zeros(
            1, max_audio_win)  ## maximum audio excerpt duration
        if not is_exist:
            return audioexcer.view(1, 1, -1)

        valid = {}
        valid['audio'] = 0
        excerptstart = self.audiodata[audioind]['starts'][start_idx]
        excerptend = self.audiodata[audioind]['ends'][end_idx]

        try:
            valid['audio'] = self.audiodata[audioind][
                'wav'][:, excerptstart:excerptend + 1].shape[1]
        except:
            pass

        audioexcer_tmp = self.audiodata[audioind][
            'wav'][:, excerptstart:excerptend + 1]

        if audioexcer_tmp.shape[1] >= max_audio_win:
            audioexcer[0, :max_audio_win] = audioexcer_tmp[0, :max_audio_win]
        else:
            audioexcer[0, :audioexcer_tmp.shape[1]] = audioexcer_tmp[0, :audioexcer_tmp.shape[1]]

        return audioexcer.view(1, 1, -1)