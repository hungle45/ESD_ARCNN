import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import torch.nn.functional as F

from .utils import (
    load_wav_to_torch,
    load_filepaths,
    wav_to_3d_mel,
)

class ESDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, speaker_id):
        
        self.sampling_rate = hparams.data.sampling_rate
        self.num_filter_bank = hparams.data.num_filter_bank
        self.list_classes = hparams.data.classes
        self.num_classes = len(self.list_classes)
        self.max_length = hparams.data.max_length
        
        self.samples = []
        for emo in self.list_classes:
            
            emo_dir = os.path.join(
                hparams.common.meta_file_folder, f'{speaker_id:04d}', emo)
            if not os.path.isdir(emo_dir):
                raise FileNotFoundError(f'Cannot find the given directory: {emo_dir}')
            
            files = os.listdir(emo_dir)
            for f in files:
                fpath = os.path.join(emo_dir, f)
                if not os.path.isfile(fpath):
                    continue
                self.samples.append((fpath, self.list_classes.index(emo)))
        
        random.seed(1234)
        random.shuffle(self.samples)

    def get_audio(self, filename):
        waveform, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        data = wav_to_3d_mel(
                    wav=waveform,
                    sampling_rate=sampling_rate,
                    num_filter=self.num_filter_bank,
                    max_length=self.max_length
                )
        return data
        
    def get_audio_label_pair(self, audiopath_label):
        audiopath, label = audiopath_label[0], audiopath_label[1]
        data = self.get_audio(audiopath)
        # label_one_hot = F.one_hot(torch.tensor(label), self.num_classes)
        return (data, label)

    def __getitem__(self, index):
        anchor = self.get_audio_label_pair(self.samples[index])
        sample_idx = np.random.randint(0, len(self.samples))
        sample = self.get_audio_label_pair(self.samples[sample_idx])
        return anchor[0], anchor[1], sample[0], sample[1] 

    def __len__(self):
        return len(self.samples)
    

class Collate:
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def __call__(self, batch):
        
        l_anchor_max, l_sample_max = 0, 0
        anchors, anchors_emo, samples, samples_emo = [], [], [], []
        
        for anchor, anchor_emo, sample, sample_emo in batch:
            
            anchors.append(anchor.permute(2, 0, 1).to(dtype=torch.float32))
            anchors_emo.append(anchor_emo)
            samples.append(sample.permute(2, 0, 1).to(dtype=torch.float32))
            samples_emo.append(sample_emo)

            l_anchor_max = max(l_anchor_max, len(anchor))
            l_sample_max = max(l_sample_max, len(sample))
        
        y = []
        
        for idx in range(len(anchors)):
            y.append(min(abs(anchors_emo[idx] - samples_emo[idx]), 1))

        anchors = torch.stack(anchors)
        samples = torch.stack(samples)
        y = torch.tensor(y)
        
        return anchors, samples, y, \
            F.one_hot(torch.tensor(anchors_emo), self.num_classes), \
            F.one_hot(torch.tensor(samples_emo), self.num_classes)