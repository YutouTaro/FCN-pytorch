# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from PIL import Image
import os
from os.path import join as pathjoin
import argparse


def pixsum(imgBW):
    imgmat = np.array(imgBW).astype(np.uint8)
    if len(imgmat.shape) == 2:  # grayscale image
        imgmat = np.expand_dims(imgmat, axis=2)
    else:
        imgmat = imgmat[:, :, :nc]
    pSum = imgmat.sum(axis=(0, 1))
    pNum = imgmat.size / nc
    return pSum, pNum

def data_util(path_file_list, dir_Img, dir_Label, dir_LabelNew, dir_ImgBW):
    ### training images, (1)link to .png file of labels, (2)convert to grayscale image (3) resize the image if --resize in input args
    # not any more [(3)create .npy file]
    fout = open(path_file_list, 'w')
    fout.write("img,label\n")
    imageNames = os.listdir(dir_Img)
    imageNames.sort()
    fileCount = 0
    pixelSum = np.zeros((nc,))
    pixelNum = 0
    for imgN in imageNames:
        if '.png' not in imgN:
            continue
        print(imgN)

        # checking label exsits
        path_label = pathjoin(dir_Label, imgN)  # label name
        if not os.path.exists(path_label):
            print("%s does not exist" % (path_label))
            continue
        else:
            # path_label_new = path_label.replace("/semantic/", "/semantic_%s/"%idx_folder)
            path_label_new = pathjoin(dir_LabelNew,imgN)
            if option.resize or not os.path.exists(path_label_new):
                imglabel = Image.open(path_label)
                if not imglabel.size == (width, height):
                    imglabel = imglabel.resize((width, height), Image.NEAREST)
                    print(" resized", end="")
                imglabel.save(path_label_new)
                print(" label saved")

        # convert rgb img to grayscale
        path_imgBW = pathjoin(dir_ImgBW, imgN)
        if option.resize or not os.path.exists(path_imgBW):
            if nc == 1:
                imgBW = Image.open(pathjoin(dir_Img, imgN)).convert('L')
            else:
                imgBW = Image.open(pathjoin(dir_Img, imgN))
            if not imgBW.size == (width, height):
                imgBW = imgBW.resize((width, height), Image.BICUBIC)
                print(" resized", end="")
            if option.calculate_mean:
                # imgmat = np.array(imgBW).astype(np.uint8)[:,:,:nc]
                # pixelSum += imgmat.sum(axis=(0,1))
                # pixelNum += imgmat.size/nc
                pSum, pNum = pixsum(imgBW)
                pixelSum += pSum
                pixelNum += pNum
            imgBW.save(path_imgBW)
            print(" image saved")
        elif option.calculate_mean:
            imgBW = Image.open(path_imgBW)
            # imgmat = np.array(imgBW).astype(np.uint8)[:, :, :nc]
            # pixelSum += imgmat.sum(axis=(0, 1))
            # pixelNum += imgmat.size / nc
            pSum, pNum = pixsum(imgBW)
            pixelSum += pSum
            pixelNum += pNum
        # create .npy file
        # path_label_npy = pathjoin(dir_trainIdx, imgN)
        # path_label_npy = path_label_npy.split('.png')[0] + '.npy'
        # if not os.path.exists(path_label_npy):
        # imgIdx = Image.open(path_label)
        # idx_mat = np.array(imgIdx).astype(np.uint8)
        # np.save(path_label_npy, idx_mat)
        # .npy file is much larger than .png file
        # modify the loader in the future to directly read .png file for labeling
        # and change the content in the csv file
        # fout.write("%s,%s\n"%(path_imgBW, path_label_npy ))
        fout.write("%s,%s\n" % (path_imgBW, path_label_new))
        fileCount += 1
    fout.close()
    print("%d valid file sets found, written in file %s" % (fileCount, path_file_list))
    if option.calculate_mean:
        print("pixel mean value = {}".format(pixelSum / pixelNum))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_dataset', '-d', type=str, required=True, help='directory to the dataset, the last folder '
                                                                         'should be data_semantics')
parser.add_argument('--calculate_mean', action='store_true', default=False,
                    help='calculate the mean value of the images')
# TODO change from boolean to self input size
parser.add_argument('--resize', action='store_true', default=False,
                    help='reshape the images and labels to predefined size')
parser.add_argument('--width', '-w', type=int, default=512,
                    help='width of the reshaped image, only useful when --resize')
parser.add_argument('--height', type=int, default=160, help='height of the reshaped image, only useful when --resize')
parser.add_argument('--channels', '-nc', type=int, required=True, help='number of output image channels', choices=[1, 3])
option = parser.parse_args()

print("=" * 10 + "directories" + "=" * 10)
# width, height = 1216, 352 # the size you want the image to be after conversion
# width, height = 512, 160 # the size you want the image to be after conversion
width, height = option.width, option.height
nc = option.channels  # number of channels
if option.resize:
    print("images and labels will be resized to (w, h)=(%d, %d)" % (width, height))
# dir_dataset = "D:\Google Drive (yutouttaro@gmail.com)\data_semantics"
# dir_dataset = "/content/drive/My Drive/data_semantics"
dir_dataset = option.dir_dataset
# input directories
dir_train = pathjoin(dir_dataset, "training")
dir_trainLabel = pathjoin(dir_train, "semantic")  # dir to semantic labels (INDEX, not color)
dir_trainImg = pathjoin(dir_train, "image_2")  # dir to the RGB images

dir_val = pathjoin(dir_dataset, "val")
dir_valLabel = pathjoin(dir_val, "semantic")
dir_valImg = pathjoin(dir_val, "image_2")

dir_testImg = pathjoin(dir_dataset, 'testing', 'image_2')  # dir to testing RGB images
input_dirs = [dir_train, dir_trainLabel, dir_trainImg, dir_val, dir_valLabel, dir_valImg, dir_testImg]
for dir in input_dirs:
    if not os.path.exists(dir):
        print("%s does not exist" % (dir))
# output directories

idx_folder = "0" if option.channels==1 else "1"
dir_trainImgBW = pathjoin(dir_train, "image_"+idx_folder)  # dir to save grayscale images
# dir_trainIdx   = pathjoin(dir_train, "label_idx")             # dir to save labeled index
dir_trainLabelNew = pathjoin(dir_train, "semantic_0")
dir_valImgBW = pathjoin(dir_val, "image_"+idx_folder)
dir_valLabelNew = pathjoin(dir_val, "semantic_0")

dir_testImgBW = pathjoin(dir_dataset, "testing", "image_"+idx_folder)  # dir to save grayscale images
output_dirs = [dir_trainImgBW, dir_trainLabelNew, dir_valImgBW, dir_valLabelNew, dir_testImgBW]
# create the directories if not exist
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("    dir created: %s" % (dir))

path_train_list = pathjoin(dir_dataset, 'train.csv')
path_val_list = pathjoin(dir_dataset, 'val.csv')
path_test_list = pathjoin(dir_dataset, 'test.csv')

for path_file_list, dir_Img, dir_Label, dir_LabelNew, dir_ImgBW in zip([path_train_list, path_val_list], [dir_trainImg, dir_valImg], [dir_trainLabel, dir_valLabel], [dir_trainLabelNew, dir_valLabelNew], [dir_trainImgBW, dir_valImgBW]):
    if "train" in path_file_list:
        print("train folder")
    else:
        print("val folder")
    data_util(path_file_list, dir_Img, dir_Label, dir_LabelNew, dir_ImgBW)

### testing images, only convert to grayscale as no labels available
fout_test = open(path_test_list, 'w')
fout_test.write("img\n")
imageNames = os.listdir(dir_testImg)
imageNames.sort()
fileCount = 0
for imgN in imageNames:
    if '.png' not in imgN:
        continue
    print(imgN)
    # convert rgb img to grayscale
    path_imgBW = pathjoin(dir_testImgBW, imgN)
    if option.resize or not os.path.exists(path_imgBW):
        if nc == 1:
            imgBW = Image.open(pathjoin(dir_testImg, imgN)).convert('L')
        else:
            imgBW = Image.open(pathjoin(dir_testImg, imgN))
        if not imgBW.size == (width, height):
            imgBW = imgBW.resize((width, height), Image.BICUBIC)
            print(" resized", end="")
        imgBW.save(path_imgBW)
        print(" image saved")
    fout_test.write("%s\n" % (path_imgBW))
    fileCount += 1
fout_test.close()
print("%d images found" % (fileCount))

print("resized train image are saved to %s" % dir_trainImgBW)
print("resized train label are saved to %s" % dir_trainLabelNew)
print("resized test image are saved to %s" % dir_testImgBW)
