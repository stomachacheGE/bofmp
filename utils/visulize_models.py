import argparse
import imp

import numpy as np
import matplotlib.pyplot as plt

from utils.occ_map_utils import show_map, plot_occ_map, plot_grid_map_hmm
from data_loader import get_iters
from test import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model predictions on data set.')
    parser.add_argument('-model', nargs='+')
    parser.add_argument('-dataset', nargs='?')
    parser.add_argument('-config_file', nargs='?')
    parser.add_argument('-show_num', nargs='?', type=int)
    parser.add_argument('-not_show_map', action='store_true', default=False)
    parser.add_argument('-show_quiver', action='store_true', default=False)

    args = vars(parser.parse_args())
    print(args)

    config_file = args['config_file']
    dataset = args['dataset']
    models_names = args['model']
    show_num = args['show_num']
    show_map_ = not args['not_show_map']
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
    trained_models_path = cf.project_folder + '/trained_models/'


    models_names_ = [trained_models_path+name for name in models_names]
    models = [get_model(name) for name in models_names_]
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
        zeros_map = np.zeros_like(map)
        preds = np.array([model(x) for model in models])
        vmax = max([np.max(preds), np.max(outputs[idx])])
        vmin = min([np.min(preds), np.min(outputs[idx])])
        if not show_quiver:
        ############################################################################
        ######## show model predictions as transitional probability heatmap ########
        ############################################################################
            m = np.random.randint(0, 4, 1)[0]
            n = np.random.randint(0, 4, 1)[0]
            up_down = [pred.reshape(4,4,32,32)[0,2,:,:] for pred in preds]
            print(up_down)
            predictions = [pred.reshape(4, 4, 32, 32)[m, :, :, :]
                           for pred in preds ]
            fig, axes_ = plt.subplots(4, len(models) + 1, sharex='all', sharey='all', figsize=(10, 10))
            for i in range(len(models)+1):
                if i == 0:
                    prob = outputs[idx].reshape(4, 4, 32, 32)[m, :, :, :]
                else:
                    prob = predictions[i-1]

                if show_map_:

                    for j in range(4):
                        plot = plt.subplot(4, len(models) + 1, j*(len(models)+1) + i + 1)
                        # axes = plot_occ_map(prob[j, :, :], map, occ_map_res=cf.resolution)
                        axes = show_map(prob[j, :, :], cmap='OrRd', resolution=cf.resolution, vmin=vmin, vmax=vmax)
                        show_map(map, resolution=cf.resolution)

                        directions_str = "p[{}|{}]".format(directions[j], directions[m])

                        if j == 0:
                            if i > 0:
                                model_name = models_names[i-1].split('/')[-1]
                                directions_str = model_name + '\n' + directions_str
                            else:
                                directions_str = 'Ground Truth' + ' \n' + directions_str

                        plot.set_title(directions_str, fontsize = 10)
                else:
                    pass
                    # plot = plt.subplot(2, len(models) + 1, i + 1)
                    # axes = plot_occ_map(prob, map, occ_map_res=cf.resolution)
                    # c_bar = plt.colorbar(axes[0], orientation='horizontal')
                    # list(axes).append(c_bar)
                    # plt.subplot(2, len(models) + 1, len(models)+1 + i + 1)
                    # axes = plot_occ_map(prob, zeros_map, occ_map_res=cf.resolution)
                    # c_bar = plt.colorbar(axes[0], orientation='horizontal')
                    # list(axes).append(c_bar)

            fig.subplots_adjust(left=0.12, bottom=0.06, right=0.7, top=0.96, wspace=0.34, hspace=0.22)
            fig.colorbar(axes, ax=axes_.ravel().tolist(), shrink=0.5)
            plt.show()
            # raw_input("Press a key to continue")
            # plt.close()
        else:
        ############################################################################
        ################### show model predictions as quiver plots #################
        ############################################################################
            fig, axes_ = plt.subplots(1, len(models) + 1, sharex='all', sharey='all', figsize=(10, 6))
            for i in range(len(preds)+1):
                plot = plt.subplot(1, len(models) + 1, i + 1)
                if i==0:
                    trans_prob = outputs[idx].reshape(4, 4, 32, 32)
                    title = 'Ground Truth'
                else:
                    trans_prob = preds[i-1].reshape(4, 4, 32, 32)
                    title = models_names[i - 1].split('/')[-1]
                axes = plot_grid_map_hmm(trans_prob, 'probs', grid_res=cf.resolution, map_=map, map_res=cf.resolution)
                plt.colorbar(axes, orientation='vertical', fraction=0.046, pad=0.04)
                plot.set_title(title, fontsize=10)
            fig.subplots_adjust(left=0.12, bottom=0.06, right=0.7, top=0.96, wspace=0.34, hspace=0.22)
            plt.show()




# fig.subplots_adjust(left=0.05, bottom=0.06, right=1, top=0.96, wspace=0, hspace=0.47)
#         plt.show()




