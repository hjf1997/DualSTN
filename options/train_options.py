#  implemented by p0werHu
# time: 5/6/2021

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        self.isTrain = True

        parser = BaseOptions.initialize(self, parser)
        # Loss visualization parameters
        parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_best', default=True, help='save the model with best results')
        parser.add_argument('--eval_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--total_iters', type=int, default=0, help='the starting iterations')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--num_train_target', type=int, default=3, help='# of nodes to infer during training')
        parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs with the initial learning rate')
        parser.add_argument('--early_stop_patience', type=int, default=10, help='trigger early stopping after this many epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=20, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        return parser
