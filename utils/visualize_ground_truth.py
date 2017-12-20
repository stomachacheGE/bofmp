import argparse
import imp

import numpy as np
import matplotlib.pyplot as plt

from utils.occ_map_utils import show_map, plot_occ_map, load_map, plot_grid_map_hmm
from data_loader import get_iters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', nargs='?')
    parser.add_argument('-config_file', nargs='?')
    parser.add_argument('-show_num', nargs='?', type=int)
    parser.add_argument('-show_quiver', action='store_true', default=False)

    args = vars(parser.parse_args())
    print(args)

    config_file = args['config_file']
    dataset = args['dataset']
    show_num = args['show_num']
    show_quiver = args['show_quiver']


    if not dataset:
        dataset = 'test'
    if not show_num:
        show_num =  10
    config_path_root = '/local/home/ful7rng/projects/transition/'
    if not config_file:
        config_path = config_path_root + 'config.py'
    else:
        config_path = config_path_root + config_file + '.py'

    cf = imp.load_source('config', config_path)

    if not show_quiver:
    ############################################################################
    ######## show ground truth as 4x4 transitional probability heatmap #########
    ############################################################################
        train_iter, val_iter, test_iter = get_iters(config_path)

        if dataset == 'train':
            iter = train_iter
        elif dataset == 'val':
            iter = val_iter
        elif dataset == 'test':
            iter = test_iter

        # randomly iterate until any batch
        for _ in range(np.random.randint(1, iter.get_n_batches(), 1)):
            iter.next()
        inputs, outputs, masks = iter.next()

        directions = cf.directions

        idxs = np.random.randint(0, cf.batch_size, (show_num,))
        for idx in idxs:

            x = inputs[idx][None, :, :, :]
            map = x[0, 0, :, :].astype('bool')
            probs = outputs[idx].reshape(4, 4, 32, 32)

            fig, axes_ = plt.subplots(4, 4, sharex='all', sharey='all', figsize=(10, 10))
            for i in range(4):
                for j in range(4):
                    plot = plt.subplot(4, 4, 4*i+j+1)
                    axes = plot_occ_map(probs[i, j, :, :], map, occ_map_res=cf.resolution)
                    directions_str = "p[{}|{}]".format(directions[j], directions[i])
                    plot.set_title(directions_str, fontsize=10)

            fig.subplots_adjust(left=0.12, bottom=0.06, right=0.7, top=0.96, wspace=0.34, hspace=0.22)
            fig.colorbar(axes[0], ax=axes_.ravel().tolist(), shrink=0.75)
            plt.show()
            # raw_input("Press a key to continue")
            # plt.close()
    else:
    ############################################################################
    #################### show ground truth as a quiver plot ####################
    ############################################################################

        # randomly sample a map
        map_name = np.random.choice(cf.training_maps+cf.test_maps, 1)[0]
        current_map_folder = cf.map_folder + '/' + map_name + '/'
        res_str = str(int(cf.resolution * 100))
        map_file = current_map_folder + 'thresholded_' + res_str + '.png'
        map_arr, _, _ = load_map(map_file)

        trans_counts_path = cf.data_folder+'/transition_maps/' + map_name + '/' \
                            + cf.algo_str + '/' + cf.algo_str + '.npy'
        trans_counts = np.load(trans_counts_path)

        input_size = cf.nn_input_size
        output_size = cf.nn_output_size
        for i in range(show_num):
            left, top = [np.random.randint(0, map_arr.shape[ix] - input_size) for ix
                         in [0, 1]]
            x_in, y_in = [slice(start, start + input_size) for start in [left, top]]
            x_out, y_out = [slice(start + int((input_size - output_size) / 2),
                                  start + int((input_size + output_size) / 2))
                            for start in [left, top]]

            input = map_arr[x_in, y_in].copy()
            trans_counts_patch = trans_counts[x_out, y_out].copy()
            plot_grid_map_hmm(trans_counts_patch, 'counts', grid_res=cf.resolution, map_=input, map_res=cf.resolution)
            plt.show()
