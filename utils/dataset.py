import os
import torch
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from .constants import *


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class AEDataset(Dataset):
    def __init__(self, root_dir, transform=data_transform):
        self.root_dir = root_dir
        self.transform = transform

        #Sort image and state paths based on frame number
        self.img_paths = sorted(
            glob(self.root_dir + '*/screenshots/*.png'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        self.ram_paths = sorted(
            glob(self.root_dir + '*/states/*.bin'),
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


class TextTrainingDataset(Dataset):
    def __init__(self, root_dir, min_interval=20, max_interval=30,
                 transform=data_transform):
        self.root_dir = root_dir
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.gap = self.max_interval - self.min_interval
        self.transform = transform
        self.length = len(glob('{}/origin/imgs/*.png'.format(self.root_dir)))

        self.max_nums = {}
        for action in sentence_dic.keys():
            self.max_nums[action] = [0] * self.length
            for i in range(self.length):
                for j in range(60, 0, -1):
                    exist = os.path.exists('{}/{}/imgs/{}.{}.png'.format(
                        self.root_dir, action, i, j))
                    if exist:
                        self.max_nums[action][i] = j
                        break

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        bool1 = np.random.choice([0, 1], p=[0.25, 0.75])
        bool2 = np.random.choice([0, 1], p=[0.33, 0.67])

        action_true, sent = random.choice(list(sentence_dic.items()))
        action_false = action_true
        while action_true == action_false:
            action_false = random.choice(list(sentence_dic.keys()))

        action = action_true if bool1 else action_false
        max_num = self.max_nums[action][idx]

        first_img_path = '{}/origin/imgs/{}.png'.format(self.root_dir, idx)
        first_img = plt.imread(first_img_path)

        #
        # if bool1:
        #     if bool2:
        #         second_num = random.randint(self.min_interval, self.max_interval)
        #     else:
        #         second_num = random.randint(1, max_num - self.gap - 1)
        #         if second_num >= self.min_interval:
        #             second_num = second_num + self.gap + 1
        # else:
        #     second_num = random.randint(1, max_num)

        if max_num == 0:
            return None
        second_num = random.randint(1, max_num)
        if second_num >= self.min_interval and second_num <= self.max_interval:
            bool2 = 1
        else:
            bool2 = 0



        second_img = plt.imread(
            '{}/{}/imgs/{}.{}.png'.format(self.root_dir, action, idx, second_num)
        )

        if self.transform:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
        first_img = first_img.to(device)
        second_img = second_img.to(device)

        sentence = prepare_sequence(sent.split(), word_to_ix)

        output = torch.zeros(2, dtype=torch.float)
        output[int(bool1 and bool2)] = 1.

        return first_img, second_img, sentence, output


###############################################################################
###############################################################################
###############################################################################


class OldTextTrainingDataset(Dataset):
    def __init__(self, root_dir, min_interval=20, max_interval=30,
                 transform=data_transform):
        self.root_dir = root_dir
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.transform = transform
        self.length = len(glob('{}/origin/imgs/*.png'.format(self.root_dir)))

        self.max_nums = {}
        for action in sentence_dic.keys():
            self.max_nums[action] = [0] * self.length
            for i in range(self.length):
                for j in range(60, 0, -1):
                    exist = os.path.exists('{}/{}/imgs/{}.{}.png'.format(
                        self.root_dir, action, i, j))
                    if exist:
                        self.max_nums[action][i] = j
                        break

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        action, sent = random.choice(list(sentence_dic.items()))
        max_num = self.max_nums[action][idx]
        if max_num <= self.max_interval:
            return None

        num = random.randint(0, max_num-self.max_interval)
        if num <= 0:
            first_img_path = '{}/origin/imgs/{}.png'.format(self.root_dir, idx)
        else:
            first_img_path =\
                '{}/{}/imgs/{}.{}.png'.format(self.root_dir, action, idx, num)

        first_img = plt.imread(first_img_path)

        second_num = num + random.randint(self.min_interval, self.max_interval)
        second_img = plt.imread(
            '{}/{}/imgs/{}.{}.png'.format(self.root_dir, action, idx, second_num)
        )

        if self.transform:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
        first_img = first_img.to(device)
        second_img = second_img.to(device)

        sentence = prepare_sequence(sent.split(), word_to_ix)

        return first_img, second_img, sentence


class TrainingDataset(Dataset):
    def __init__(self, root_dir, sentence_num=10, interval=30,
                 transform=data_transform):
        self.root_dir = root_dir
        self.sentence_num = sentence_num
        self.interval = interval
        self.transform = transform

        #Sort image and state paths based on frame number
        self.img_paths = sorted(
            glob(self.root_dir + 'screenshots/*'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        self.ram_paths = sorted(
            glob(self.root_dir + 'states/*'),
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
            glob(self.root_dir + 'screenshots/*'),
            key=lambda i: int(os.path.basename(i).split('.')[0]))
        self.ram_paths = sorted(
            glob(self.root_dir + 'states/*'),
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
