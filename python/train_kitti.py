from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from kitti_loader import kittiDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from os.path import join as pathjion
import datetime
from shutil import copyfile
import math
import argparse

n_class    = 34

parser =argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'camvid', 'cityscape'], help='name of the dataset')
parser.add_argument('--dir_dataset', '-d', type=str, required=True, help='directory to the dataset, the last folder should be data_semantics')
parser.add_argument('--batchsize', type=int  , default=6   , help='input batch size')
parser.add_argument('--epochs'   , type=int  , default=500 , help='number of epochs to train')
parser.add_argument('--lr'       , type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum' , type=float, default=0   , help='momentum')
parser.add_argument('--w_decay'  , type=float, default=1e-5, help='weight decay')
parser.add_argument('--step_size', type=int  , default=50  , help='step size')
parser.add_argument('--gamma'    , type=float, default=0.5 , help='gamma')


parser.add_argument('--isCrop'        , action='store_true', default=False, help='crop the image?')
parser.add_argument('--flip_rate'     , type=float         , default=0.5  , help='flip rate')
parser.add_argument('--new_height'    , type=int           , default=370  , help='height after crop')
parser.add_argument('--new_width'     , type=int           , default=1224 , help='width after crop')

parser.add_argument('--continue_train', action='store_true', default=False, help='[train]is continue training by loading a model parameter?')
parser.add_argument('--which_folder'  , type=str           , default=''   , help='the folder to load the parameter for test/continue train')
parser.add_argument('--which_epoch'   , type=int           , default=0    , help='the epoch to load for test/continue training')
parser.add_argument('--isTest'        , action='store_true', default=False, help='is test?')

option = parser.parse_args()

# configs    = "FCNs-BCEWithLogits\nbatch size: {}\nepoch: {}\nRMSprop scheduler step size: {}\ngamma: {}\nlearn rate: {}\nmomentum: {}\nw_decay: {}".format(
#     option.batchsize, option.epochs, option.step_size, option.gamma, option.lr, option.momentum, option.w_decay)
# print("Configs:", configs)

if option.continue_train and option.isTest: # cannot be True at the same time
    print("error with the train/test config!")
    quit()

if option.continue_train or option.isTest:
    # TODO get the configs as input
    epoch_count = 500
    save_path = "/content/drive/My Drive/models/net-%s/net_%03d.pth" % ("200519-094259", epoch_count)
    option.lr *= math.pow(option.w_decay, int(epoch_count/30))
else:
    epoch_count = 0

dir_root = option.dir_dataset
path_train_file = pathjion(dir_root, 'train.csv')
#create dir for saving model parameters later on
dir_model = pathjion(dir_root, "models")
if not os.path.exists(dir_model):
    os.makedirs(dir_model)
if option.isTest or option.continue_train:
    dir_model = pathjion(dir_model, option.which_folder)
    if not os.path.exists(dir_model):
        print("%s does not exist! Check your input argument of \"--which_folder\"" % dir_model)
        quit()
else: # create a new folder to save
    timeNow = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(seconds=28800)))
    model_folder = timeNow.strftime("net-%y%m%d-%H%M%S")
    dir_model = pathjion(dir_model, model_folder)
    os.makedirs(dir_model)

# save the config file
print('='*20)
with open(pathjion(dir_model, "config.txt"), 'w') as fout:
    tempStr = 'FCNs, BCEWithLogits, RMSprop scheduler'
    fout.write(tempStr+'\n')
    print(tempStr)
    for k, v in sorted(vars(option).items()):
        tempStr = '%s: %s' % (str(k), str(v))
        fout.write(tempStr+'\n')
        print(tempStr)
print('='*20)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

# TODO select models from option
train_data = kittiDataset(option= option, csv_file=path_train_file, isTrain = True, n_class=n_class)