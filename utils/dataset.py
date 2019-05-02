import os
import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .constants import *


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

class TrainingDataset(Dataset):
    def __init__(self, root_dir, sentence_num=10, interval=30,
                 transform=data_transform):
        self.root_dir = root_dir
        self.sentence_num = sentence_num
        self.interval = interval
        self.transform = transform

        #Sort image and state paths based on frame number
        self.img_paths = sorted(
            glob(root_dir + 'screenshots/*'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        self.ram_paths = sorted(
            glob(root_dir + 'states/*'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        assert len(self.img_paths) == len(self.ram_paths), "Paths don't match"

        self.frame_nums = []
        self.none_nums = set()
        sentences = []
        for i, path in enumerate(self.img_paths):
            file_name = os.path.basename(path).split('.')
            self.frame_nums.append(int(file_name[0]))
            if len(file_name) == 2:
                self.none_nums.add(i)
                sentences.append(torch.zeros(1, dtype=torch.long))
            else:
                sentences.append(prepare_sequence(
                    sentence_dic[file_name[1]].split(), word_to_ix))
        self.sentences = pad_sequence(sentences, batch_first=True)


    def __len__(self):
        return len(self.img_paths) - self.interval

    def get_data(self, idx):
        img_path = self.img_paths[idx]
        img = plt.imread(img_path)
        if self.transform:
            img = self.transform(img)
        img = img.to(device)

        ram_path = self.ram_paths[idx]
        ram = np.fromfile(ram_path, dtype=np.uint8)
        ram = torch.from_numpy(ram).to(device, dtype=torch.long)

        return img, ram

    def __getitem__(self, idx1):
        img1, ram1 = self.get_data(idx1)

        idx2 = idx1 + self.interval
        img2, ram2 = self.get_data(idx2)

        sentence = torch.zeros(8, dtype=torch.long, device=device)
        if idx2 not in self.none_nums and\
            self.frame_nums[idx2] - self.frame_nums[idx1] <= self.interval + 10:
            sentence = self.sentences[idx2]

        return img1, ram1, img2, ram2, sentence


class TestingDataset(Dataset):
    def __init__(self, root_dir, interval=30, get_ram=False,
                 transform=data_transform):
        self.root_dir = root_dir
        self.interval = interval
        self.get_ram = get_ram
        self.transform = transform

        #Sort image and state paths based on frame number
        self.img_paths = sorted(
            glob(root_dir + 'screenshots/*'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        self.ram_paths = sorted(
            glob(root_dir + 'states/*'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        assert len(self.img_paths) == len(self.ram_paths), "Paths don't match"


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = plt.imread(img_path)
        if self.transform:
            img = self.transform(img)
        img = img.to(device)

        num = int(os.path.basename(img_path).split('.')[0])
        action = os.path.basename(img_path).split('.')[1]

        ram = 0
        if self.get_ram:
            ram_path = self.ram_paths[idx]
            ram = np.fromfile(ram_path, dtype=np.uint8)
            ram = torch.from_numpy(ram).to(device, dtype=torch.long)

            num2 = int(os.path.basename(ram_path).split('.')[0])
            assert num == num2, "Image path doesn't match with ram path"

        return img, ram, (num, action)


class AEDataset(Dataset):
    def __init__(self, root_dir, transform=data_transform):
        self.root_dir = root_dir
        self.transform = transform

        #Sort image and state paths based on frame number
        self.img_paths = sorted(
            glob(root_dir + '*/screenshots/*.png'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        self.ram_paths = sorted(
            glob(root_dir + '*/states/*.bin'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        assert len(self.img_paths) == len(self.ram_paths), "Paths don't match"


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = plt.imread(img_path)
        if self.transform:
            img = self.transform(img)
        img = img.to(device)

        ram_path = self.ram_paths[idx]
        ram = np.fromfile(ram_path, dtype=np.uint8)
        ram = torch.from_numpy(ram).to(device, dtype=torch.long)

        return img, ram
