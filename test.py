

import os
import time
import sys

if sys.version_info[0] < 3:
    import imp
else:
    from importlib.machinery import SourceFileLoader

import numpy as np
import theano
from lasagne.layers import get_output

from data_loader import get_iters
from evaluation import eval_dataset, get_best_pars
from utils import binary_kl, binary_sym_kl, binary_rev_kl, categorical_kl_div

def get_module23(module, pck):
    if sys.version_info[0] < 3:
        return imp.load_source(module, pck)
    else:
        return SourceFileLoader(module, pck).load_module()

def test(pred_fn, loss_fn, mode='val', plots=False):

    _, val_iter, test_iter = get_iters(batch_size=8)

    if mode == 'val':
        iter = val_iter
    elif mode == 'test':
        iter = test_iter

    # test with best parameters
    loss, time = eval_dataset(
        [lambda sample: pred_fn(sample[None, None, :, :])],
        loss_fn, iter, plots=plots)[0]

    print('loss of {:.3f}, average runtime: {:.5f} s'.format(loss, time))
    return loss, time


def init_model(net, weight_file):

    ###############
    #  Load data  #
    ###############

    print('-' * 75)
    # Load config file


    ###################
    #  Compile model  #
    ###################

    # Print summary
    net.summary()
    net.restore(weight_file)

    # Compile test functions
    prediction = get_output(net.output_layer, deterministic=True,
                            batch_norm_use_averages=False)

    # compile prediction function
    start_time_compilation = time.time()
    print('Compiling function')
    pred_function = theano.function([net.input_var], prediction)
    print('Compilation took {:.3f} seconds'.format(
        time.time() - start_time_compilation))

    return pred_function


def get_best_model(exp_folder, k_best=0):
    best = get_best_pars(exp_folder + 'summary.csv', k_best)
    best_folder = exp_folder + str(int(best[0]))
    return get_model(best_folder)


def get_model(exp_folder):
    cf_file = exp_folder + '/cf.py'
    weight_file = exp_folder + '/model.npz'

    # cf = imp.load_source('cf', cf_file)
    cf = get_module23("module", cf_file)

    return init_model(cf.net, weight_file)


if __name__ == '__main__':
    exp_folder = os.path.dirname(__file__) + '/trained_models/22_previous_models_cat_ce/CNN_26_big'
    print(exp_folder)
    loss_fn = categorical_kl_div

    pred_fn = get_model(exp_folder)
    loss, time_ = test(pred_fn, loss_fn, mode='test', plots=True)

    # losss = []
    # times = []
    # for i in range(10):
    #     pred_fn = get_model(exp_folder + str(i))
    #     loss, time_ = test(pred_fn, loss_fn, mode='test')
    #
    #     losss.append(loss)
    #     times.append(time_)
    # print('total loss: {} pm {}'.format(np.mean(losss), np.std(losss)))
    # print('total run time: {} pm {}'.format(np.mean(times), np.std(times)))


