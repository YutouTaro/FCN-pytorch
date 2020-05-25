from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from kitti_loader import kittiDataset, show_batch

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
parser.add_argument('--model'     , type=str  , default='fcns', choices=['fcn32s', 'fcn16s', 'fcn8s','fcns'], help="the strucuture of the model")
parser.add_argument('--batch_size', type=int  , default=4   , help='input batch size')
parser.add_argument('--epochs'    , type=int  , default=500 , help='number of epochs to train')
parser.add_argument('--lr'        , type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum'  , type=float, default=0   , help='momentum')
parser.add_argument('--w_decay'   , type=float, default=1e-5, help='weight decay')
parser.add_argument('--step_size' , type=int  , default=50  , help='step size')
parser.add_argument('--gamma'     , type=float, default=0.5 , help='gamma')


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
dir_root = option.dir_dataset
path_train_file = pathjion(dir_root, 'train.csv')
path_test_file = pathjion(dir_root, 'test.csv')
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

if option.continue_train and option.isTest: # cannot be True at the same time
    print("error with the train/test config!")
    quit()

if option.continue_train or option.isTest:
    # TODO get the configs as input
    epoch_count = 500
    save_path = pathjion(dir_root, "models", option.which_folder, "net_%03d.pth"%(epoch_count) )
    # save_path = "%s/models/net-%s/net_%03d.pth" % (dir_root, option.which_folder, epoch_count)
    option.lr *= math.pow(option.w_decay, int(epoch_count/50))
else:
    epoch_count = 0

if option.model == 'fcns':
    model = FCNs
elif option.model == 'fcn8s':
    model = FCN8s
elif option.model == 'fcn16s':
    model = FCN16s
elif option.model == 'fcn32s':
    model = FCN32s
else:
    print("input model name does not recognised!")
    quit()


# save the config file
print('='*20)
config_fileName = "config_test.txt" if option.isTest else "config_train.txt"
with open(pathjion(dir_model, config_fileName), 'w') as fout:
    tempStr = '%s, BCEWithLogits, RMSprop scheduler' % option.model
    fout.write(tempStr+'\n')
    print(tempStr)
    for k, v in sorted(vars(option).items()):
        tempStr = '%s: %s' % (str(k), str(v))
        fout.write(tempStr+'\n')
        print(tempStr)
print('='*20)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
if use_gpu:
    print("cuda detected: {}".format(num_gpu))

# TODO select models from option
train_data = kittiDataset(option= option, csv_file=path_train_file, isTrain=True, n_class=n_class)
test_data   = kittiDataset(option= option, csv_file=path_test_file, isTrain=False, n_class=n_class)

train_loader = DataLoader(train_data, batch_size=option.batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=1, num_workers=8)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = model(pretrained_net=vgg_model, n_class=n_class)

if option.isTest or option.continue_train:
    fcn_model.load_state_dict(torch.load(save_path))
    print("loaded parameters from %s" % (save_path))

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading in %.1f sec" % (time.time() - ts))
    device = torch.device("cuda:0")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=option.lr, momentum=option.momentum, weight_decay=option.w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=option.step_size, gamma=option.gamma)

# create dir for score
# score_dir = os.path.join("scores", configs)
dir_score = os.path.join(dir_model, "scores")
if not os.path.exists(dir_score):
    os.makedirs(dir_score)
IU_scores    = np.zeros((option.epochs, n_class))
pixel_scores = np.zeros(option.epochs)

def train():
    for epoch in range(epoch_count+1, option.epochs+1):
        timestart_epoch = time.time()
        timestart_iters = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = Variable(batch['X']).to(device)
                labels = Variable(batch['Y']).to(device)
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print("\tepoch: %d, iter %d, loss: %.3f, learn_rate: %.7f, %.2f sec" % (
                epoch, iter, loss.data, lr, time.time() - timestart_iters))
                timestart_iters = time.time()

        scheduler.step()

        model_name = pathjion(dir_model, "net_latest.pth")
        torch.save(fcn_model.module.state_dict(), model_name)
        lr = optimizer.param_groups[0]['lr']
        print("Epoch %d, loss: %.3f, learn_rate: %.7f, %.2f sec" % (epoch, loss.data, lr, time.time() - timestart_epoch))
        if epoch % 10 == 0:
            net_name = pathjion(dir_model, "net_%03d.pth"%(epoch))
            copyfile(model_name, net_name)

def test(epoch=0):
    fcn_model.eval()
    for iter, batch in enumerate(test_loader):
        timeIter = time.time()
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)
        plt.figure()
        show_batch(batch)
        plt.axis('off')
        plt.ioff()
        plt.show()
        print("\tepoch: %d, iter: %d, %.2f sec" % (epoch, iter, time.time() - timeIter))

# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    if option.isTest:
        test(option.which_epoch)
    else:
        train()