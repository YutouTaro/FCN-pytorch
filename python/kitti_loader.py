# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

# import imageio
from PIL import Image

means = np.array([99.703])

class kittiDataset(Dataset):
    def __init__(self, option, csv_file, isTrain, n_class=34):
        self.data = pd.read_csv(csv_file)
        self.means = means
        self.n_class = n_class

        if isTrain:
            self.crop = option.isCrop
            self.flip_rate = 0.5
            self.new_h = option.new_height
            self.new_w = option.new_width
        else:
            # self.crop = crop
            self.flip_rate = option.flip_rate
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx,0]
        img = Image.open(img_name)
        img = np.array(img).astype(np.uint8)[:,:,0]
        label_name = self.data.iloc[idx,1]
        # label = np.load(label_name) # old format, read .npy file which converted in kitti_utils
        imglabel = Image.open(label_name)
        label = np.array(imglabel).astype(np.uint8)[:,:,0]
        assert img.shape == label.shape

        if self.crop:
            h, w  = img.size
            top   = random.randint(0, h-self.new_h)
            left  = random.randint(0, w-self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img = np.fliplr(img)
            label = np.fliplr(label)

        # reduce mean
        img -= means[0]
        img /= 255.

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    # img_batch[:,1,...].add_(means[1])
    # img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    # plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
    plt.imshow(grid.numpy()[::-1])
    plt.title('Batch from dataloader')


if __name__ == "__main__":
    path_train_file = r'D:\Google Drive (yutouttaro@gmail.com)\data_semantics\train.csv'
    train_data = kittiDataset(csv_file=path_train_file, isTrain=True)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size(), batch['Y'].size())

        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break