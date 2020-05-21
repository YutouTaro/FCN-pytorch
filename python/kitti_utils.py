# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from PIL import Image
import os
from os.path import join as pathjoin

print("="*10 + "directories" + "="*10)
# dir_dataset = "D:\Google Drive (yutouttaro@gmail.com)\data_semantics"
dir_dataset = "/content/drive/My Drive/data_semantics"

# input directories
dir_train      = pathjoin(dir_dataset, "training")
dir_trainLabel = pathjoin(dir_train, "semantic")              # dir to semantic labels (INDEX, not color)
dir_trainImg   = pathjoin(dir_train, "image_2")               # dir to the RGB images

dir_testImg    = pathjoin(dir_dataset, 'testing', 'image_2')  # dir to testing RGB images
input_dirs = [dir_train, dir_trainLabel,dir_trainImg,  dir_testImg]
for dir in input_dirs:
    if not os.path.exists(dir):
        print("%s does not exist" % (dir))
# output directories

dir_trainImgBW = pathjoin(dir_train, "image_0")               # dir to save grayscale images
dir_trainIdx   = pathjoin(dir_train, "label_idx")             # dir to save labeled index

dir_testImgBW  = pathjoin(dir_dataset, "testing", "image_0")  # dir to save grayscale images

output_dirs = [dir_trainImgBW, dir_trainIdx, dir_testImgBW]
# create the directories if not exist
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("    dir created: %s" % (dir))

path_train_list = pathjoin(dir_dataset, 'train.csv')
# path_test_list = pathjoin(dir_dataset, 'test.csv')

width, height = 1242, 375 # the size you want the image to be after conversion # TODO not used right now

# Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]

### training images, (1)link to label .png file, (2)create .npy file and (3)convert to grayscale image
fout = open(path_train_list, 'w')
fout.write("img,label\n")
imageNames = os.listdir(dir_trainImg)
imageNames.sort()
fileCount = 0
for imgN in imageNames:
    if '.png' not in imgN:
        continue

    # checking label exsits
    path_label = pathjoin(dir_trainLabel, imgN) # label name
    if not os.path.exists(path_label):
        print("%s does not exist" % (path_label))
        continue

    # convert rgb img to grayscale
    path_imgBW = pathjoin(dir_trainImgBW, imgN)
    if not os.path.exists(path_imgBW):
        imgBW = Image.open(pathjoin(dir_trainImg,imgN)).convert('LA')
        # if not imgBW.size == (width, height):
        #     imgBW = imgBW.resize((width, height), Image.NEAREST)
        imgBW.save(path_imgBW)

    # create .npy file
    path_label_npy = pathjoin(dir_trainIdx, imgN)
    path_label_npy = path_label_npy.split('.png')[0] + '.npy'
    if not os.path.exists(path_label_npy):
        imgIdx = Image.open(path_label)
        idx_mat = np.array(imgIdx).astype(np.uint8)
        np.save(path_label_npy, idx_mat)
    # .npy file is much larger than .png file
    # TODO modify the loader in the future to directly read .png file for labeling
    # TODO and change the content in the csv file
    fout.write("%s,%s\n"%(path_imgBW, path_label_npy ))
    fileCount += 1
fout.close()
print("%d valid file sets found, written in file %s" % (fileCount, path_train_list))

### testing images, only convert to grayscale as no labels available
# fout = open(path_test_list, 'w')
# fout.write("img,label\n")
imageNames = os.listdir(dir_testImg)
imageNames.sort()
fileCount = 0
for imgN in imageNames:
    if '.png' not in imgN:
        continue
    # convert rgb img to grayscale
    path_imgBW = pathjoin(dir_testImgBW, imgN)
    if not os.path.exists(path_imgBW):
        imgBW = Image.open(pathjoin(dir_testImg, imgN)).convert('LA')
        # if not imgBW.size == (width, height):
        #     imgBW = imgBW.resize((width, height), Image.NEAREST)
        imgBW.save(path_imgBW)
    # fout.write("%s,%s\n" % (path_imgBW, path_label_npy))
    fileCount += 1
# fout.close()
print("%d images found" % (fileCount))