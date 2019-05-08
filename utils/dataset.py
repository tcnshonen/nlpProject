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
    if len(batch) == 0:
        return None
    return default_collate(batch)


class MixDataset(Dataset):
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
        #bool1 = np.random.choice([0, 1], p=[0.25, 0.75])
        #bool2 = np.random.choice([0, 1], p=[0.3, 0.7])
        bool1 = np.random.choice([0, 1], p=[0.3, 0.7])

        action_true, sent = random.choice(list(sentence_dic.items()))
        # action_false = action_true
        # while action_true == action_false:
        #     action_false = random.choice(list(sentence_dic.keys()))
        if 'left' in sent:
            action_false = random.choice(list(sentence_dic.keys())[3:])
        else:
            action_false = random.choice(list(sentence_dic.keys())[:3])

        action = action_true if bool1 else action_false
        max_num = self.max_nums[action][idx]

        first_img_path = '{}/origin/imgs/{}.png'.format(self.root_dir, idx)
        first_img = plt.imread(first_img_path)
        first_ram_path = '{}/origin/states/{}.bin'.format(self.root_dir, idx)
        first_ram = np.fromfile(first_ram_path, dtype=np.uint8)
        first_ram = torch.from_numpy(first_ram).to(device, dtype=torch.long)


        if max_num == 0:
            return None

        second_num = random.randint(1, max_num)

        # if bool1:
        #     if bool2:
        #         if max_num >= self.max_interval:
        #             second_num = random.randint(self.min_interval, self.max_interval)
        #         elif max_num > self.min_interval:
        #             second_num = random.randint(self.min_interval, max_num)
        #         else:
        #             return None
        #     else:
        #         second_num = random.randint(1, max_num - self.gap - 1)
        #         if second_num >= self.min_interval:
        #             second_num = second_num + self.gap + 1
        # else:
        #     second_num = random.randint(1, max_num)

        second_img_path = '{}/{}/imgs/{}.{}.png'.format(
            self.root_dir, action, idx, second_num)
        second_img = plt.imread(second_img_path)
        second_ram_path = '{}/{}/states/{}.{}.bin'.format(
            self.root_dir, action, idx, second_num)
        second_ram = np.fromfile(second_ram_path, dtype=np.uint8)
        second_ram = torch.from_numpy(second_ram).to(device, dtype=torch.long)


        if self.transform:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
        first_img = first_img.to(device)
        second_img = second_img.to(device)

        sentence = prepare_sequence(sent.split(), word_to_ix)

        cls = torch.zeros(2, dtype=torch.float).to(device)
        cls[bool1] = 1.

        return first_img, first_ram, second_img, second_ram, sentence, cls


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

        output = torch.zeros(2, dtype=torch.float).to(device)
        output[int(bool1 and bool2)] = 1.

        return first_img, second_img, sentence, output


###############################################################################
###############################################################################
###############################################################################


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
