import os
import cv2
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import time
from PIL import Image
import glob
import sys 
import random
import librosa
from scipy import signal
import lmdb

class FlickrTestDataset(Dataset):
    def __init__(self, data_list, data_path):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        self.base_path = data_path
        self.data_list = [line.strip() for line in lines]
        self.imgSize = 224
        self._init_transform()
    
    def __len__(self):
        return len(self.data_list)
    
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([
            transforms.Resize(self.imgSize, Image.BICUBIC),
            transforms.CenterCrop(self.imgSize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])            
    
    def __getitem__(self, idx):
        filename = self.data_list[idx]
        img_path = os.path.join(self.base_path, "images", filename + ".jpg")
        audio_path = os.path.join(self.base_path, "audios", filename + ".wav")

        samples, sr = librosa.load(audio_path, sr=24000)

        resamples = samples[len(samples) // 2 - int(sr * 1.5): len(samples) // 2 + int(sr * 1.5)]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, sr, nperseg=512, noverlap=274)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)

        img = Image.open(img_path)
        img = self.img_transform(img)
        return img, spectrogram, filename

class LmdbDataset(Dataset):
    def __init__(self, args, logger, mode='train'):
        data_list = args.train_list if self.mode == 'train' else args.val_list
        with open(data_list, 'r') as f:
            lines = f.readlines()
        self.data_list = [line.strip() for line in lines]

        self.audio_env = lmdb.open(os.path.join(args.base_path, "{}_audios.lmdb".format(mode)), readonly=True, lock=False, readahead=False, meminit=False)
        self.image_env = lmdb.open(os.path.join(args.base_path, "{}_images.lmdb".format(mode)), readonly=True, lock=False, readahead=False, meminit=False)

        self.imgSize = args.image_size
        self.sr = args.sample_rate

        self.logger = logger
        self.mode = mode
        self._init_transform()
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_frame(self, key, size=(3, 256, 256)):
        with self.image_env.begin(write=False) as image_txn:
            buf = image_txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        C, H, W = size
        img = img_flat.reshape(H, W, C)
        img = img[:, :, [2, 1, 0]]
        img = Image.fromarray(img, mode='RGB')
        return img
    
    def _load_audio(self, key, size=(257, 301)):
        with self.audio_env.begin(write=False) as audio_txn:
            buf = audio_txn.get(key.encode('ascii'))
        audio = np.frombuffer(buf, dtype=np.float32)
        audio = np.copy(audio)
        if audio.shape[0] < self.sr * 10:
            n = int(self.sr * 10 / audio.shape[0]) + 1
            samples = np.tile(samples, n)
        return audio
    
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.imgSize, self.imgSize), Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(self.args.mean, self.args.std)])
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])            
    
    def __getitem__(self, idx):
        filename = self.data_list[idx]
        audio = self._load_audio(filename)

        # sample the 3s audio clip in the middle and the corresponding frame by default
        resamples = audio[len(audio) // 2 - int(self.sr * 1.5): len(audio) // 2 + int(self.sr * 1.5)]
        img_key = filename + '/007.jpg'

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, self.sr, nperseg=512, noverlap=274)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)

        img = self._load_frame(img_key)
        img = self.img_transform(img)
        return img, spectrogram, filename
    
class AVDataset(Dataset):
    def __init__(self, args, logger, mode='train'):

        self.audio_path = os.path.join(args.base_path, 'audios')
        self.frame_path = os.path.join(args.base_path, 'video_frames')

        self.imgSize = args.image_size
        self.sr = args.sample_rate

        self.args = args
        self.mode = mode

        #  Retrieve list of audio and video files, remove files that are non-exist
        self._load_samples(args)
        logger.info("Totally {} samples in the {} set".format(len(self.video_files), mode))

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.imgSize, self.imgSize), Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(self.args.mean, self.args.std)])
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])            

    def _load_samples(self, args):
        data_list = args.train_list if self.mode == 'train' else args.val_list
        with open(data_list, 'r') as f:
            lines = f.readlines()
        self.video_files = []
        for line in lines:
            video_name = line.strip()
            audio_file = os.path.join(self.audio_path, video_name + self.args.audio_suffix)
            image_file = os.path.join(self.frame_path, video_name, '007.jpg')
            if os.path.exists(audio_file) and os.path.exists(image_file):
                self.video_files.append(video_name)

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        filename = self.video_files[idx]
        # Image
        frame = self.img_transform(self._load_frame(os.path.join(self.frame_path, filename, '007.jpg')))
        # Audio
        samples, _ = librosa.load(os.path.join(self.audio_path, filename + self.args.audio_suffix), sr=self.sr)
        resamples = samples[len(samples) // 2 - int(self.sr * 1.5): len(samples) // 2 + int(self.sr * 1.5)]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, self.sr, nperseg=512,noverlap=274)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
        return frame, spectrogram, filename