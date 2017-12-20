
from __future__ import print_function


import os
import shutil
import sys
import time
import argparse
from datetime import datetime
import imp
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.regularization import regularize_network_params
from lasagne.layers import get_output
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compile.nanguardmode import NanGuardMode

import fileinput

from data_loader import get_iters
from test import test, get_model

from utils import categorical_kl_div, random_search, binary_sym_kl, binary_kl


def batch_loop(iterator, f, pred_f, epoch, phase, history, no_mask, conditional):
    """ Loop on the batches """

    n_batches = iterator.get_n_batches()
    # n_imgs = 0.

    for i in range(n_batches):
        X, Y, masks = iterator.next()
        # batch_size = X.shape[0]
        # batch_size, n_channels, n_rows, n_cols = Y.shape

        # n_imgs += batch_size

        loss = f(X, Y, masks)

        pred = pred_f(X)

        if i == 0:
            ce_tot = loss
            kl_tot = categorical_kl_div(pred, Y, masks, no_mask=no_mask, conditional=conditional)
        else:
            ce_tot += loss
            kl_tot += categorical_kl_div(pred, Y, masks, no_mask=no_mask, conditional=conditional)

        # # Progression bar ( < 74 characters)
        sys.stdout.write('\rEpoch {} : [{} : {}%]'.format(epoch, phase, int(100. * (i + 1) / n_batches)))
        sys.stdout.flush()

    history[phase]['kl-div'].append(kl_tot / n_batches)
    history[phase]['loss'].append(ce_tot / n_batches)

    return history


def train(cf):

    ###############
    #  load data  #
    ###############

    print('-' * 75)
    print('Loading data')
    #TODO ; prepare a public version of the data loader
    train_iter, val_iter, test_iter = get_iters(cf.config_path)

    print('Number of images : train : {}, val : {}, test : {}'.format(
        train_iter.get_n_samples(), val_iter.get_n_samples(), test_iter.get_n_samples()))

    ###################
    #   Build model   #
    ###################

    # Build model and display summary
    net = cf.net
    net.summary()

    # Restore
    if hasattr(cf, 'pretrained_model'):
        print('Using a pretrained model : {}'.format(cf.pretrained_model))
        net.restore(cf.pretrained_model)

    # Compile functions
    print('Compilation starts at ' + str(datetime.now()).split('.')[0])
    params = lasagne.layers.get_all_params(net.output_layer, trainable=True)
    lr_shared = theano.shared(np.array(cf.learning_rate, dtype='float32'))
    lr_decay = np.array(cf.lr_sched_decay, dtype='float32')

    # Create loss and metrics
    for key in ['train', 'valid']:

        # LOSS
        pred = get_output(net.output_layer, deterministic=key=='valid',
                          batch_norm_update_averages=False, batch_norm_use_averages=False)
        masks = T.tensor4('target_var', dtype='float32')  # masks
        norm_fac = cf.output_norm_fac
        loss = \
            cf.loss_function(norm_fac * pred, norm_fac * net.target_var, masks,
                             no_mask=cf.no_mask, conditional=cf.conditional_prob)

        pred_fn = theano.function([net.input_var], pred, on_unused_input='warn')

        if cf.weight_decay:
            weightsl2 = regularize_network_params(net.output_layer, lasagne.regularization.l2)
            loss += cf.weight_decay * weightsl2

        # COMPILE
        start_time_compilation = time.time()
        if key == 'train':
            updates = cf.optimizer(loss, params, learning_rate=lr_shared)
            train_fn = theano.function([net.input_var, net.target_var, masks], loss,
                                       updates=updates,
                                       on_unused_input='warn')
        else:
            val_fn = theano.function([net.input_var, net.target_var, masks], loss,
                                     on_unused_input='warn')

        print('{} compilation took {:.3f} seconds'.format(key, time.time() - start_time_compilation))

    ###################
    #    Main loops   #
    ###################

    # metric's sauce
    init_history = lambda: {'loss': [], 'kl-div':[]}
    history = {'train': init_history(), 'val': init_history(), 'test': init_history()}
    patience = 0
    best_kl_div = np.inf
    best_epoch = 0

    if hasattr(cf, 'pretrained_model'):
        print('Validation score before training')
        print(batch_loop(val_iter, val_fn, 0, 'val', {'val': init_history()}))

    # Training main loop
    print('-' * 30)
    print('Training starts at ' + str(datetime.now()).split('.')[0])
    print('-' * 30)

    for epoch in range(cf.num_epochs):

        # Train
        start_time_train = time.time()
        history = batch_loop(train_iter, train_fn, pred_fn, epoch, 'train',
                             history, cf.no_mask, cf.conditional_prob)
        # Validation
        start_time_valid = time.time()
        history = batch_loop(val_iter, val_fn, pred_fn, epoch, 'val', history, cf.no_mask, cf.conditional_prob)

        # Print
        out_str = \
            '\r\x1b[2 Epoch {} took {}+{} sec. ' \
            'kl-div={:.5f} | loss = {:.5f} || kl-div={:.5f} | loss = {:.5f}'.format(
                epoch, int(start_time_valid - start_time_train),
                int(time.time() - start_time_valid),
                history['train']['kl-div'][-1],
                history['train']['loss'][-1],
                history['val']['kl-div'][-1],
                history['val']['loss'][-1])

        # Monitoring loss
        if history['val']['kl-div'][-1] < best_kl_div:
            out_str += ' (BEST)'
            best_kl_div = history['val']['kl-div'][-1]
            best_epoch = epoch
            patience = 0
            net.save(os.path.join(cf.savepath, 'model.npz'))
        else:
            patience += 1

        print(out_str)

        np.savez(os.path.join(cf.savepath, 'errors.npz'), metrics=history, best_epoch=best_epoch)

        # Learning rate scheduler
        lr_shared.set_value(lr_shared.get_value() * lr_decay)

        # Finish training if patience has expired or max nber of epochs reached
        if patience == cf.max_patience or epoch == cf.num_epochs - 1:
            # Load best model weights
            net.restore(os.path.join(cf.savepath, 'model.npz'))

            # Test
            print('Training ends\nTest')
            if test_iter.get_n_samples() == 0:
                print('No test set')
            else:
                history = batch_loop(test_iter, val_fn, pred_fn, epoch, 'test',
                                     history, no_mask=cf.no_mask, conditional=cf.conditional_prob)

                print ('Average kl-divergence of test images = {:.5f}'.format(
                    history['test']['kl-div'][-1]))

                np.savez(os.path.join(cf.savepath, 'errors.npz'), metrics=history, best_epoch=best_epoch)

            # Exit
            return


def initiate_training(cf):

    # Seed : to make experiments reproductible, use deterministic convolution in CuDNN with THEANO_FLAGS
    np.random.seed(cf.seed)
    theano.tensor.shared_randomstreams.RandomStreams(cf.seed)

    if not os.path.exists(cf.savepath):
        os.makedirs(cf.savepath)
    else:
        stop = raw_input('\033[93m The following folder already exists {}. '
                         'Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf.savepath))
        if stop == 'n':
            return

    print('-' * 75)
    print('Config\n')
    print('Local saving directory : ' + cf.savepath)
    print('Model path : ' + cf.model_path)

    # We also copy the model and the training scipt to reproduce exactly the experiments

    shutil.copy(os.path.join(os.path.dirname(__file__), 'train.py'),
                os.path.join(cf.savepath, 'train.py'))
    shutil.copy(cf.model_path, os.path.join(cf.savepath, 'model.py'))
    shutil.copy(cf.config_path, os.path.join(cf.savepath, 'cf.py'))
    shutil.copy(cf.project_folder+'/metrics.py', os.path.join(cf.savepath, 'metrics.py'))


    # Train
    train(cf)

def train_with_pars(conf_file, loss_fn, exp_folder, exp_name, **parameters):
    f = fileinput.input(files=(conf_file), inplace=True)

    for line in f:
        write_str = line
        for par_name in parameters:
            assign_str = par_name + ' = '
            if line.startswith(assign_str):
                write_str = assign_str + str(parameters[par_name]) + '\n'
        print(write_str, end='')

    f.close()
    print('', end='\n')

    cf = imp.load_source('config', conf_file)
    cf.savepath = exp_folder + exp_name
    cf.config_path = conf_file

    initiate_training(cf)

    pred_fn = get_model(exp_folder + exp_name)
    return test(pred_fn, loss_fn, mode='val')

def random_search_cnn(conf_file, loss_fn, exp_folder):
    record_file = exp_folder + 'summary.csv'

    # perform random search over hyper parameters
    num_pts = 1000
    random_search(lambda *args, **kwargs:
                  train_with_pars(conf_file, loss_fn, exp_folder, *args, **kwargs),
                  record_file, num_pts,
                  lambda: str(np.random.randint(0, 1e9)),  # experiment id
                  learning_rate=lambda: 10 ** np.random.uniform(-6, -3),
                  lr_sched_decay=lambda: 1 - 10 ** np.random.uniform(-3, -1),
                  weight_decay=lambda: 10 ** np.random.uniform(-7, -3),
                  output_norm_fac=lambda: 1,
                  n_filters_first_conv=lambda: np.random.choice([4, 8, 16]),
                  n_pool=lambda: np.random.choice([0, 1, 2]),
                  growth_rate=lambda: np.random.choice([6, 8, 10, 12]),
                  n_layers_per_block=lambda: np.random.choice([1, 2, 3, 4, 5]),
                  dropout_p=lambda: np.random.uniform(0.1, 0.5))

    # random_search(lambda *args, **kwargs:
    #               train_with_pars(conf_file, loss_fn, exp_folder, *args, **kwargs),
    #               record_file, num_pts,
    #               lambda: str(np.random.randint(0, 100000)),  # experiment id
    #               learning_rate=lambda: 1.3296543010712417e-4,
    #               lr_sched_decay=lambda: 0.9975539976766103,
    #               weight_decay=lambda: 0.0005246603324480762,
    #               output_norm_fac=lambda: 1,
    #               n_filters_first_conv=lambda: 16,
    #               n_pool=lambda: 2,
    #               growth_rate=lambda: 10,
    #               n_layers_per_block=lambda: 2,
    #               dropout_p=lambda: 0.36742755957117645)



if __name__ == '__main__':
    # module_path = os.path.dirname(__file__) + '/'
    # conf_file = module_path + 'trained_models/22_previous_models_cat_ce/CNN-26_big.py'
    # exp_folder = module_path + 'trained_models/22_previous_models_cat_ce/'
    # loss_fn = categorical_kl_div
    #
    # # random_search_cnn(conf_file, loss_fn, exp_folder)
    # print(conf_file)
    # cf = imp.load_source('config', conf_file)
    #
    # cf.savepath = exp_folder + 'CNN_26_big_fu'
    # cf.config_path = conf_file
    # initiate_training(cf)

    config_path_root = '/local/home/ful7rng/projects/transition/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', nargs='?')

    args = vars(parser.parse_args())
    print(args)

    config_name = args['config']

    if not config_name:
        config_path = config_path_root + 'config.py'
    else:
        config_path = config_path_root + config_name + '.py'

    cf = imp.load_source('config', config_path)
    print(cf.algo_str)
    cf.config_path = config_path
    initiate_training(cf)




