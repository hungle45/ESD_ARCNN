import time
import os
import random

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from datasets import Dataset, Audio

from .utils import (
    load_wav_to_torch,
    load_filepaths,
    wav_to_3d_mel,
    load_metadata
)

class ESDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, speaker_id, test_size=None, seed=45):
        self.path = hparams.common.meta_file_folder
        self.sampling_rate = hparams.data.sampling_rate
        self.num_filter_bank = hparams.data.num_filter_bank
        self.max_length = hparams.data.max_length

        self.speaker_id = speaker_id
        self.test_size = test_size
        self.seed = seed
        
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
        return {
            'audio': {
                'path': filename,
                'array': data
            }
        }
        
    def get_data(self):
        metadata = load_metadata(self.path, self.speaker_id)
        data = Dataset.from_dict(metadata) \
            .map(lambda sample: self.get_audio(sample['audio']))
        data = data.class_encode_column('emotion')
        
        if self.test_size is not None:
            data = data.train_test_split(test_size=self.test_size, seed=self.seed)
        return data
    

class DataCollator():
    def __init__(self, num_labels):
        self.num_labels = num_labels
    
    def __call__(self, data):
        audio_arrays = []
        ground_truth = []
        emotion_dict = {}
        for idx, sample in enumerate(data):
            audio_arrays.append(torch.Tensor(sample['audio']['array']))
            ground_truth.append(sample['emotion'])
            current_emotion = str(sample['emotion'])
            if current_emotion not in emotion_dict:
                emotion_dict[current_emotion] = [idx]
            else:
                emotion_dict[current_emotion].append(idx)
                
        anchor_audios = []
        negative_audios = []
        targets = []
        negative_emo = []
        for idx, target_emotion in enumerate(ground_truth):
            target = random.randint(0, 1)
            siamese_index = idx
            negative_sample_emo = target_emotion
            if target == 1:
                siamese_index = random.choice(emotion_dict[str(target_emotion)])
            else:
                # Get all type of emotion existed
                exist_emo = list(emotion_dict.keys())
                # Remove target emotion if it existed in list
                if str(target_emotion) in exist_emo:
                    exist_emo.remove(str(target_emotion)) # List of negative emotion
                
                # Case: Only exist target emotion
                if len(exist_emo) == 0:
                    target = 1
                    siamese_index = random.choice(emotion_dict[str(target_emotion)])
                else:
                    negative_sample_emo = random.choice(exist_emo)
                    siamese_index = random.choice(emotion_dict[negative_sample_emo])
            anchor_audios.append(audio_arrays[idx])
            negative_audios.append(audio_arrays[siamese_index])
            targets.append(target)
            negative_emo.append(int(negative_sample_emo))
            
        return (torch.stack(anchor_audios),
                torch.stack(negative_audios),
                torch.tensor(targets),
                F.one_hot(torch.tensor(ground_truth), self.num_labels),
                F.one_hot(torch.tensor(negative_emo), self.num_labels)
            ) 