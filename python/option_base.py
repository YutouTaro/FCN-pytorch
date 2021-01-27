import argparse
import os
import torch
import datetime

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dir_dataset', '-d', type=str, required=True,
                            help='directory to the dataset, the last folder should be data_semantics')
        self.parser.add_argument('--type', '-t', type=str, required=True, choices=['train', 'val', 'test'],
                                 help="type of the running")
        self.parser.add_argument('--model', type=str, default='fcns', choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcns'],
                            help="the strucuture of the model")
        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self.parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--momentum', type=float, default=0, help='momentum')
        self.parser.add_argument('--w_decay', type=float, default=1e-5, help='weight decay')
        self.parser.add_argument('--step_size', type=int, default=50, help='step size')
        self.parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
        self.parser.add_argument('--channels', '-nc', type=int, required=True,
                            help='number of image channels', choices=[1, 3])

        self.parser.add_argument('--isCrop', action='store_true', default=False, help='crop the image?')
        self.parser.add_argument('--flip_rate', type=float, default=0.5, help='flip rate')
        self.parser.add_argument('--new_height', type=int, default=370, help='height after crop')
        self.parser.add_argument('--new_width', type=int, default=1224, help='width after crop')

        self.initialized = True

    def parse(self):
        print('test')
        if not self.initialized:
            self.initialize()
        self.option = self.parser.parse_args()
        # self.option.type = self.type

        args = vars(self.option)

        #save the options
        dir_model = os.path.join(self.option.dir_dataset, "models")
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        if self.option.type != 'train' or self.option.continue_train:
            dir_model = os.path.join(dir_model, self.option.which_folder)
            if not os.path.exists(dir_model):
                print("%s does not exist! Check your input argument of \"--which_folder\"" % dir_model)
                quit()
        else:  # create a new folder to save
            timeNow = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(seconds=28800)))
            model_folder = timeNow.strftime("net-%y%m%d-%H%M%S")
            self.option.which_folder = model_folder
            dir_model = os.path.join(dir_model, model_folder)
            os.makedirs(dir_model)
        config_fileName = "config_%s.txt" % self.option.type

        print('------------ Options -------------')

        with open(os.path.join(dir_model, config_fileName), 'w') as fout:
            for k,v in sorted(args.items()):
                tempStr = '%s: %s' % (str(k), str(v))
                fout.write(tempStr + '\n')
                print(tempStr)
        print('-------------- End ----------------')


        return self.option