
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time

from data_loader import get_iters, get_iters_map
from utils import random_search
from utils.occ_map_utils import display_occ_map, display_occ_maps

map_str = '100_7'

def test_model(pred_fn, loss_fn, train_loss_fn=None, verbose=False,
               plots=False):
    if train_loss_fn is None:
        train_loss_fn = loss_fn

    print('Testing model: {} with loss {}'.format(
        pred_fn.__name__, loss_fn.__name__))

    record_file = pred_fn.__name__ + '_' + train_loss_fn.__name__ + \
                  '_' + map_str + '.csv'

    dat = np.loadtxt(record_file, delimiter=',\t')
    best = dat[np.argmin(dat[:, -2])]

    print('best parameter set: {}'.format(np.array(best[:-2])))

    _, val_iter, test_iter = get_iters(batch_size=8)
    iter_ = test_iter

    # test with best parameters
    loss, time = eval_dataset(
        [lambda *x: pred_fn(*(list(x) + list(best[:-2])))],
        loss_fn, iter_, verbose=verbose, plots=plots)[0]

    print('loss of {:.3f}, average runtime: {:.5f} s'.format(loss, time))

def train_model(pred_fn, loss_fn, *pred_fn_args, **pred_fn_kwargs):
    whole_map = False
    num_pts = 100
    if whole_map:
        _, val_iter, test_iter = get_iters_map()
    else:
        _, val_iter, test_iter = get_iters()
    iter = val_iter

    record_file = pred_fn.__name__ + '_' + loss_fn.__name__ + \
                  '_' + map_str + '.csv'

    random_search(lambda *x, **y:
                  tuple(eval_dataset([lambda *u:
                               pred_fn(*(u + x), **y)],loss_fn, iter)[0]),
                  record_file, num_pts, *pred_fn_args, **pred_fn_kwargs)


def eval_dataset(pred_fns, loss_fn, iter_, verbose=False, plots=False,
                 plot_folder = None):

    n_batches = iter_.get_n_batches()
    n_samples = n_batches * iter_.batch_size

    total_losss = np.zeros_like(pred_fns)
    total_times = np.zeros_like(pred_fns)
    for batch_ix in range(n_batches):
        batch = iter_.next()
        for sample_ix in range(iter_.batch_size):


            def index_batch(batch, index):
                return batch[index, 0]

            inputs = index_batch(batch[0], sample_ix)
            labels = index_batch(batch[1], sample_ix)
            masks = index_batch(batch[2], sample_ix)

            # calculate predictions
            preds = np.empty([len(pred_fns), inputs.shape[0], inputs.shape[1]])
            losss = np.empty([len(pred_fns), inputs.shape[0], inputs.shape[1]])
            for pred_fn_ix, pred_fn in enumerate(pred_fns):
                start_time = time()
                pred = pred_fn(inputs).reshape(labels.shape)
                # exclude predictions on walls
                pred[inputs > 0] = 0
                pred /= pred.sum()
                preds[pred_fn_ix] = pred
                run_time = time() - start_time
                total_times[pred_fn_ix] += run_time
                # calculate loss
                loss = loss_fn(pred, labels, masks, elementwise=True)
                losss[pred_fn_ix] = loss
                total_losss[pred_fn_ix] += loss.sum()

                if verbose:
                    print("{}/{}: {}: losses {:.4f}, time: {"
                          ":}".format(iter_.batch_size * batch_ix + sample_ix,
                                      n_samples, pred_fn.__name__,
                                      loss.sum(), run_time))
            if plots:
                if sample_ix % 8 == 0:
                    map_arr = inputs.astype(bool)
                    display_occ_maps(
                        np.concatenate([labels[None, ...], preds], axis=0)[
                            None, ...],
                        map_arr[None, :],
                        titles=['ground_truth'] + [fn.__name__ for fn in
                                                   pred_fns],
                        labels=np.hstack([[0.0], losss.sum(axis=(-1, -2))])[None, :],
                        resolution=0.2)

                    if plot_folder is not None:
                        plt.savefig(
                            os.path.join(plot_folder, 'scene_{}.svg'.format(
                                batch_ix)),
                            format='svg')
                        plt.close()
                    else:
                        plt.show()


    avg_losss = total_losss / n_samples
    avg_times = total_times / n_samples
    return np.stack([avg_losss, avg_times]).T


def get_best_pars(file, k_best=0):
    dat = np.loadtxt(file, delimiter=',\t')
    dat_sorted = dat[dat[:, -2].argsort()]
    best = dat_sorted[k_best]

    print('best parameter set: {}'.format(np.array(best[:-2])))
    return best[:-2]