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
        img = np.array(img).astype(np.float32)[:,:,0]
        label_name = self.data.iloc[idx,1]
        # label = np.load(label_name) # old format, read .npy file which converted in kitti_utils
        imglabel = Image.open(label_name)
        label = np.array(imglabel).astype(np.uint8)
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
    gridnp = grid.numpy()[::-1]
    b,h,w = gridnp.size
    gridnp = gridnp.reshape((b,1,h,w))
    plt.imshow(gridnp)
    plt.title('Batch from dataloader')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'camvid', 'cityscape'], help='name of the dataset')
    parser.add_argument('--dir_dataset', '-d', type=str, required=True,
                        help='directory to the dataset, the last folder should be data_semantics')
    parser.add_argument('--batchsize', type=int, default=6, help='input batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0, help='momentum')
    parser.add_argument('--w_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--step_size', type=int, default=50, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    parser.add_argument('--isCrop', action='store_true', default=False, help='crop the image?')
    parser.add_argument('--flip_rate', type=float, default=0.5, help='flip rate')
    parser.add_argument('--new_height', type=int, default=375, help='height after crop')
    parser.add_argument('--new_width', type=int, default=1242, help='width after crop')

    parser.add_argument('--continue_train', action='store_true', default=False,
                        help='[train]is continue training by loading a model parameter?')
    parser.add_argument('--which_folder', type=str, default='',
                        help='the folder to load the parameter for test/continue train')
    parser.add_argument('--which_epoch', type=int, default=0, help='the epoch to load for test/continue training')
    parser.add_argument('--isTest', action='store_true', default=False, help='is test?')

    option = parser.parse_args()
    dir_root = option.dir_dataset
    path_train_file = os.path.join(dir_root, 'train.csv')
    train_data = kittiDataset(option=option, csv_file=path_train_file, isTrain=True)

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