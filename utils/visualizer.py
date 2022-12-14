import os
import sys
import time


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.metrics_name = os.path.join(opt.checkpoints_dir, opt.name, 'metrics.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        with open(self.metrics_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Metrics (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_metrics(self, epoch, iters, metrics, t_val):
        """print current losses on console; also save the losses to the disk
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_val)
        for k, v in metrics.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
        with open(self.metrics_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message