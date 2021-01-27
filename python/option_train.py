# import sys
# sys.path.append('D:\GitHub_repository\FCN-pytorch\python\*')
from option_base import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--continue_train', action='store_true', default=False,
                            help='Is continue training by loading a model parameter?')
        self.parser.add_argument('--which_folder', type=str, default='',
                            help='the folder to save&load the parameter for test/continue train')
        self.parser.add_argument('--which_epoch', type=int, default=0,
                            help='the epoch to load for continue training, or the starting epoch for testing')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')

        # self.isTrain = True

if __name__ == "__main__":
    to = TrainOptions()
    opt = TrainOptions.parse(to)
