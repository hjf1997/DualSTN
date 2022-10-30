# implemented by p0werHu
# time 5/6/2021

import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)

    # evaluation
    test_opt = TestOptions().parse()   # get testing options
    test_dataset = create_dataset(test_opt)  # create a dataset given opt.dataset_mode and other options
    test_dataset_size = len(test_dataset)    # get the number of samples in the dataset.
    print('The number of testing samples = %d' % test_dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save and plots
    total_iters = 0                # the total number of training iterations
    best_metric = None  # best metric
    early_stop_trigger = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        model.train()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:   # display images on visdom and save images to
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

            iter_data_time = time.time()

        # evaluation on test dataset, we didn't use the validation set in this project
        if epoch % opt.eval_epoch_freq == 0:
            model.eval()
            test_start_time = time.time()
            for i, data in enumerate(test_dataset):
                model.set_input(data)
                model.test()
                model.cache_results()  # store current batch results
            t_test = time.time() - test_start_time
            model.compute_metrics()  # compute metrics
            metrics = model.get_current_metrics
            visualizer.print_current_metrics(epoch, total_iters, metrics, t_test)

            if opt.save_best and (best_metric is None or best_metric['RMSE'] > metrics['RMSE']):
                print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                best_metric = metrics.copy()
                model.save_networks('best')
                model.save_data()
                early_stop_trigger = 0
            else:
                early_stop_trigger += 1
                if early_stop_trigger >= opt.early_stop_patience:
                    print('early stop at epoch %d, iters %d' % (epoch, total_iters))
                    break

            model.clear_cache()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates in the beginning of every epoc

    visualizer.print_current_metrics(-1, total_iters, best_metric, 0)
    print('End of training')