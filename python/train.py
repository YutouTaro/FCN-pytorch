# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from Cityscapes_loader import CityScapesDataset
from CamVid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
import datetime
from shutil import copyfile
import math

n_class    = 20

batch_size = 6
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5

configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler_step{}_gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

continueTrain = False
isTest = True
if continueTrain or isTest:
    # epoch_count = 250 # opt.epoch_count
    epoch_count = 500 # TODO # opt.epoch_count
    # save_path = "/content/drive/My Drive/models/net-%s/net_%03d.pth" % ("200516-225630", epoch_count)
    save_path = "/content/drive/My Drive/models/net-%s/net_%03d.pth" % ("200519-094259", epoch_count)
    lr *= math.pow(w_decay, int(epoch_count/30))
else:
    epoch_count = 0

if sys.argv[1] == 'CamVid':
    root_dir   = "CamVid/"
else: # cityscapes
    root_dir   = "/content/drive/My Drive/"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = os.path.join(root_dir, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#TODO #if continueTrain:
if not isTest:
    timeNow = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(seconds=28800)))
    model_folder = timeNow.strftime("net-%y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, model_folder)
    os.makedirs(model_path)
    with open( os.path.join(model_path, "config.txt"), "w" ) as fout:
        fout.write(configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

if sys.argv[1] == 'CamVid':
    train_data = CamVidDataset(csv_file=train_file, phase='train')
else:
    train_data = CityScapesDataset(csv_file=train_file, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

if sys.argv[1] == 'CamVid':
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
else:
    val_data = CityScapesDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
if continueTrain or isTest:
    fcn_model.load_state_dict(torch.load(save_path))
    print("loaded parameters from %s" % (save_path))

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
    device = torch.device("cuda:0")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
# score_dir = os.path.join("scores", configs)
score_dir = os.path.join(model_dir, "scores")
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epoch_count+1, epochs+1):
        # scheduler.step() # since torch 1.1.0, `lr_scheduler.step()` must be called after `optimizer.step()`, so move to the end

        ts = time.time()
        timeTrain = time.time()
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
                # print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0])) ### PyTorch>=0.5, the index of 0-dim tensor is invalid
                # print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data))
                lr = optimizer.param_groups[0]['lr']
                print("\tepoch: %d, iter %d, loss: %.3f, learn_rate: %.7f, %.2f sec" % (epoch, iter, loss.data, lr, time.time() - timeTrain))
                timeTrain = time.time()

        scheduler.step()


        model_name = os.path.join(model_path, "net_latest.pth")
        # torch.save(fcn_model, model_name)
        # torch.save(fcn_model.cpu().state_dict(), model_name)
        torch.save(fcn_model.module.state_dict(), model_name)
        lr = optimizer.param_groups[0]['lr']
        print("Epoch %d, loss: %.3f, learn_rate: %.7f, %.2f sec" % (epoch, loss.data, lr, time.time() - ts))
        if epoch % 10 == 0:
            net_name = os.path.join(model_path, "net_%03d.pth"%(epoch))
            copyfile(model_name, net_name)

        # torch.save(fcn_model, model_path)

        ##### commented the val in each epoch
        # val(epoch)


def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
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

        print("\tepoch: %d, iter: %d, %.2f sec" % (epoch, iter, time.time()-timeIter))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch-1] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch-1] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


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
    # val(0)  # show the accuracy before training
    if isTest:
        val(epochs)
    else:
        train()
