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




class kittiDataset(Dataset):
    def __init__(self, option, csv_file, withLabel, n_class=34):
        self.data = pd.read_csv(csv_file)
        self.n_class = n_class
        self.withLabel = withLabel
        if option.channels == 1:
            # self.means = np.repeat(99.21219586, 3) / 255.  # np.array([99.21219586])/255.
            self.means = np.repeat(99.23180176, 3) / 255.  # np.array([99.21219586])/255.
        else:
            self.means = np.array([96.6757915,  101.60559698,  97.83071057]) / 255.
        if withLabel:
            self.crop = option.isCrop
            self.flip_rate = 0.5
            self.new_h = option.new_height
            self.new_w = option.new_width
        else:
            self.crop = False
            self.flip_rate = 0
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx,0]
        img = Image.open(img_name)
        img = np.array(img).astype(np.float32)
        if len(img.shape) == 2: # grayscale image
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        num_channel = img.shape[2]
        if num_channel in [2, 4]:
            img = img[...,:-1]
        if num_channel == 1:
            img = np.repeat(img, 3, axis=2)
        if self.withLabel:
            label_name = self.data.iloc[idx,1]
            # label = np.load(label_name) # old format, read .npy file which converted in kitti_utils
            imglabel = Image.open(label_name)
            label = np.array(imglabel).astype(np.uint8)
            assert img.shape[:2] == label.shape[:2]

        if self.crop:
            h, w  = img.size
            top   = random.randint(0, h-self.new_h)
            left  = random.randint(0, w-self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            if self.withLabel:
                label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img = np.fliplr(img)
            if self.withLabel:
                label = np.fliplr(label)

        # reduce mean
        img = np.transpose(img, (2, 0, 1)) / 255.
        for i in range(img.shape[0]):
            img[i] -= self.means[i]
        # img[0] -= means[0]
        # img[1] -= means[1]
        # img[2] -= means[2]
        # img = img[np.newaxis, ...]
        # img = np.stack((img, img, img))

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        if self.withLabel:
            label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        if self.withLabel:
            h, w = label.size()
            target = torch.zeros(self.n_class, h, w)
            for c in range(self.n_class):
                target[c][label == c] = 1

            sample = {'X': img, 'Y': target, 'l': label}
        else:
            sample = {'X': img}
        return sample

def show_batch(batch):
    img_batch = batch['X']
    img_batch[:,0,...].add_(means[0])
    img_batch[:,1,...].add_(means[1])
    img_batch[:,2,...].add_(means[2])
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    # plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
    gridnp = grid.numpy()[::-1].transpose((1, 2, 0))
    # gridnp = gridnp[:,:,0]
    # b,h,w = gridnp.size
    # gridnp = gridnp.reshape((b,1,h,w))
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
    withLabel = not option.isTest
    if withLabel:
        path_train_file = os.path.join(dir_root, 'train.csv')
        train_data = kittiDataset(option=option, csv_file=path_train_file, withLabel=True)
    else:
        path_test_file = os.path.join(dir_root, 'test.csv')
        test_data = kittiDataset(option=option, csv_file=path_test_file, withLabel=False)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        if withLabel:
            sample = train_data[i]
            print(i, sample['X'].size(), sample['Y'].size())
        else:
            sample = test_data[i]
            print(i, sample['X'].size())

    if withLabel:
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        if withLabel:
            print(i, batch['X'].size(), batch['Y'].size())
        else:
            print(i, batch['X'].size())
        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break