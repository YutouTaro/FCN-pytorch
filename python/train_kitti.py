from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os, sys
if os.name == 'nt': # windows10
    # path_append = 'D:\GitHub_repository\FCN-pytorch\python\*'
    sys.path.append('D:\GitHub_repository\FCN-pytorch\python\*')
# elif os.name == 'posix': # colab
#     path_append = '/content/FCN-pytorch/FCN-pytorch/python/*'
# sys.path.append(path_append)
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from kitti_loader import kittiDataset, show_batch
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from os.path import join as pathjoin
import datetime
from shutil import copyfile
import math
import argparse
from inspect import currentframe
from collections import namedtuple
from collections import Counter
from collections import OrderedDict

def __line__():
    cf = currentframe()
    return cf.f_back.f_lineno

n_class = 34

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'camvid', 'cityscape'], help='name of the dataset')
parser.add_argument('--dir_dataset', '-d', type=str, required=True,
                    help='directory to the dataset, the last folder should be data_semantics')
parser.add_argument('--model', type=str, default='fcns', choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcns'],
                    help="the strucuture of the model")
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0, help='momentum')
parser.add_argument('--w_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--step_size', type=int, default=50, help='step size')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--channels', '-nc', type=int, required=True,
                    help='number of image channels', choices=[1, 3])

parser.add_argument('--isCrop', action='store_true', default=False, help='crop the image?')
parser.add_argument('--flip_rate', type=float, default=0.5, help='flip rate')
parser.add_argument('--new_height', type=int, default=370, help='height after crop')
parser.add_argument('--new_width', type=int, default=1224, help='width after crop')

parser.add_argument('--continue_train', action='store_true', default=False,
                    help='[train]is continue training by loading a model parameter?')
parser.add_argument('--which_folder', type=str, default='',
                    help='the folder to save&load the parameter for test/continue train')
parser.add_argument('--which_epoch', type=int, default=0, help='the epoch to load for continue training, or the starting epoch for testing')
parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--isTest', action='store_true', default=False, help='is test?')
parser.add_argument('--isVal', action='store_true', default=False, help='is val?')

option = parser.parse_args()

# configs    = "FCNs-BCEWithLogits\nbatch size: {}\nepoch: {}\nRMSprop scheduler step size: {}\ngamma: {}\nlearn rate: {}\nmomentum: {}\nw_decay: {}".format(
#     option.batchsize, option.epochs, option.step_size, option.gamma, option.lr, option.momentum, option.w_decay)
# print("Configs:", configs)
dir_root = option.dir_dataset
path_train_file = pathjoin(dir_root, 'train.csv')
path_val_file = pathjoin(dir_root, 'val.csv')
path_test_file = pathjoin(dir_root, 'test.csv')
# create dir for saving model parameters later on
dir_model = pathjoin(dir_root, "models")
if not os.path.exists(dir_model):
    os.makedirs(dir_model)
if option.isTest or option.continue_train or option.isVal:
    dir_model = pathjoin(dir_model, option.which_folder)
    if not os.path.exists(dir_model):
        print("%s does not exist! Check your input argument of \"--which_folder\"" % dir_model)
        quit()
else:  # create a new folder to save
    timeNow = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(seconds=28800)))
    model_folder = timeNow.strftime("net-%y%m%d-%H%M%S")
    option.which_folder = model_folder
    dir_model = pathjoin(dir_model, model_folder)
    os.makedirs(dir_model)

if sum(map(bool, [option.continue_train, option.isTest, option.isVal])) > 1:  # cannot be True at the same time
    print("error with the train/test config!")
    quit()

if option.continue_train or option.isTest:# or option.isVal:
    epoch_count = option.which_epoch
    save_path = pathjoin(dir_root, "models", option.which_folder, "net_%03d.pth" % (epoch_count))
    # save_path = "%s/models/net-%s/net_%03d.pth" % (dir_root, option.which_folder, epoch_count)
    option.lr *= math.pow(option.w_decay, int(epoch_count / 50))
else:
    epoch_count = 0

if option.isTest or option.isVal:
    option.batch_size = 1

# select model from option
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
print('=' * 20)
if option.isTest:
    config_fileName = "config_test.txt"
elif option.isVal:
    config_fileName = "config_val.txt"
else:
    config_fileName = "config_train.txt"

with open(pathjoin(dir_model, config_fileName), 'w') as fout:
    tempStr = '%s, BCEWithLogits, RMSprop scheduler' % option.model
    fout.write(tempStr + '\n')
    print(tempStr)
    for k, v in sorted(vars(option).items()):
        tempStr = '%s: %s' % (str(k), str(v))
        fout.write(tempStr + '\n')
        print(tempStr)
print('=' * 20)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
if use_gpu:
    print("cuda detected: {}".format(num_gpu))

train_data = kittiDataset(option=option, csv_file=path_train_file, withLabel=True,  n_class=n_class)
val_data   = kittiDataset(option=option, csv_file=path_val_file,   withLabel=True,  n_class=n_class)
test_data  = kittiDataset(option=option, csv_file=path_test_file,  withLabel=False, n_class=n_class)

train_loader = DataLoader(train_data, batch_size=option.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)
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
# dir_score = os.path.join(dir_model, "scores")
# if not os.path.exists(dir_score):
#     os.makedirs(dir_score)
# IU_scores = np.zeros((option.epochs, n_class))
# pixel_scores = np.zeros(option.epochs)
def idx2clr():
    # index2color
    Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]
    index2color = {}
    index2color[0] = (0, 0, 0)
    for obj in labels:
        # if obj.ignoreInEval:
        #     continue
        # idx = obj.id  # trainId
        # label = obj.name
        # color = obj.color
        # color2index[color] = idx
        index2color[obj.id] = obj.color
        # print("{}:{}".format(color, idx))
    return index2color


def train():
    # fcn_model.train() # better result without this line (tbd)
    flog = open(pathjoin(dir_model, "train_log.txt"), 'w')
    for epoch in range(epoch_count + 1, option.epochs + 1):
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

        model_name = pathjoin(dir_model, "net_latest.pth")
        torch.save(fcn_model.module.state_dict(), model_name)
        lr = optimizer.param_groups[0]['lr']
        logStr = "Epoch %d, loss: %.3f, learn_rate: %.7f, %.2f sec" % (epoch, loss.data, lr, time.time() - timestart_epoch)
        print(logStr)
        flog.write(logStr+'\n')
        if epoch % option.save_epoch_freq == 0:
            net_name = pathjoin(dir_model, "net_%03d.pth" % (epoch))
            copyfile(model_name, net_name)
    flog.close()

def val():
    dir_valScore = pathjoin(dir_root, 'val', 'scores')
    if not os.path.exists(dir_valScore):
        os.makedirs(dir_valScore)
    dir_valScore = pathjoin(dir_valScore, option.which_folder)
    if not os.path.exists(dir_valScore):
        os.makedirs(dir_valScore)
        print("%s created." % dir_valScore)
    else:
        print("%s exists." % dir_valScore)

    valEpochs = range(option.which_epoch, option.epochs+1, option.save_epoch_freq)
    num_saved_epochs = len(valEpochs)
    IU_scores = np.zeros((num_saved_epochs,n_class))
    pixel_scores = np.zeros(num_saved_epochs)


    for i, epoch in zip(range(num_saved_epochs), valEpochs):
        total_ious = []
        pixel_accs = []
        model_path = pathjoin(dir_root, "models", option.which_folder, "net_%03d.pth" % (epoch))
        assert os.path.exists(model_path)
        # print("model path: %s" % (model_path))
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module." not in k:
                k = "module." + k
            new_state_dict[k] = v
        fcn_model.load_state_dict(new_state_dict)
        fcn_model.eval()
        for iter, batch in enumerate(val_loader):
            timeIter = time.time()
            if use_gpu:
                inputs = Variable(batch['X'].cuda())
            else:
                inputs = Variable(batch['X'])

            output = fcn_model(inputs)
            output = output.data.cpu().numpy()

            N, _, h, w = output.shape
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

            target = batch['l'].cpu().numpy().reshape(N, h, w)
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))

            print("\tepoch: %d, iter: %d, %.2f sec" % (epoch, iter, time.time() - timeIter))

        # Calculate average IoU
        total_ious = np.array(total_ious).T  # n_class * val_len
        ious = np.nanmean(total_ious, axis=1)
        pixel_accs = np.array(pixel_accs).mean()
        print("epoch {}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
        IU_scores[i] = ious
        pixel_scores[i] = pixel_accs

    np.save(os.path.join(dir_valScore, "meanIU"), IU_scores)
    np.save(os.path.join(dir_valScore, "meanPixel"), pixel_scores)


def test(epoch=0):

    index2color = idx2clr()

    fcn_model.eval()
    dir_predict = pathjoin(dir_root, 'testing', 'predict')
    if not os.path.exists(dir_predict):
        os.makedirs(dir_predict)
    dir_predict = pathjoin(dir_predict, option.which_folder)
    if not os.path.exists(dir_predict):
        os.makedirs(dir_predict)
        print("%s created." % dir_predict)
    else:
        print("%s exists." % dir_predict)

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
        # plt.figure()
        imgout = pred.transpose((1, 2, 0))
        imgout = imgout[:, :, 0]
        # plt.imshow(imgout)
        # plt.axis('off')
        # plt.ioff()
        # plt.show()
        print("\tepoch: %d, iter: %d, %.2f sec" % (epoch, iter, time.time() - timeIter))
        # im = Image.fromarray((imgout/n_class*255).astype(np.uint8))
        path_pred_np = pathjoin(dir_predict, "pred_%03d.npy"%iter)
        # im.save(path_pred_img)
        np.save(path_pred_np, imgout)
        # threshold = imgout.size * 0
        imgrgb_np = np.zeros((h,w,3))
        # imgrgb_np = np.stack((imgrgb_np, imgrgb_np, imgrgb_np))
        for idx, num in Counter(imgout.flatten()).items():
            # if num <= threshold:
            # prednp[prednp==idx] = 0
            # if num >= threshold:
            imgrgb_np[imgout == idx] = index2color[idx]
        # print(sorted(Counter(imgout.flatten()).items()))
        imgrgb = Image.fromarray(imgrgb_np.astype(np.uint8))
        path_pred_img = pathjoin(dir_predict, "pred_%03d.png"%iter)
        imgrgb.save(path_pred_img)
        # break


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        if pred_inds.sum() == 0 and target_inds.sum() == 0:
            ious.append(float('nan'))
            continue
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        # if union == 0:
        #     ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        # else:
        assert(union > 0)
        ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    # total = (target == target).sum()
    # total = target.size
    return correct / target.size


if __name__ == "__main__":
    if option.isVal:
        val()
    elif option.isTest:
        test(option.which_epoch)
    else:
        train()