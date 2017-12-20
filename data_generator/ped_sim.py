import os
import re
import imp

import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from astar_ped_sim.astar_traj_generator import \
    sample_trajectories, rescale_trajectory

from utils import get_npy, ensure_dir

from utils.occ_map_utils import load_map, free_space, plot_trajectories


def calc_trajectories(map_arr, num_trajs, mode='transverse', resolution=1.0, min_dist=10,
                      max_dist=20, diagonal=False, verbose=False):
    """ Sample trajectories and rescale them. """
    raw_trajectories = sample_trajectories(map_arr, num_trajs,
                                           mode=mode,
                                           min_dist=min_dist / resolution,
                                           max_dist=max_dist / resolution,
                                           diagonal=diagonal,
                                           verbose=verbose)
    #TODO: rescaling can be done by array manipulation
    trajectories = []
    for ix, trajectory in enumerate(raw_trajectories):
        if verbose and ix % 1000 == 0:
            print('rescaling trajectories: {}/{}'.format(
                ix + 1, len(raw_trajectories)))
        trajectory = rescale_trajectory(trajectory, resolution)
        trajectories.append(trajectory)
    return trajectories

def shift_trajectories(trajs, offset):
    """ Shift trajectories by an offset. """
    trajs_tmp = []
    for traj in trajs:
        traj[:, -2:] += offset[None, :]
        trajs_tmp += [traj]
    return trajs_tmp

def generate_trajectories(map_str, config_path, plot=False):
    """
    Generate trajectories for map specified as map_str.
    """
    cf = imp.load_source('config', config_path)
    mode = cf.trajectory_sampling_mode

    traj_folder = cf.data_folder + '/trajectories'
    res_str = str(int(cf.resolution * 100))
    mean_path_length = np.mean([cf.min_traj_length, cf.max_traj_length])

    print('Generate trajectories for map {}'.format(map_str))
    current_map_folder = cf.map_folder + '/' + map_str + '/'
    map_file = current_map_folder + 'thresholded_' + res_str + '.png'
    map_arr, _, _ = load_map(map_file)
    average_free_space = np.mean(free_space(map_arr))

    # sample paths
    file_name = os.path.join(
        traj_folder, map_str, cf.algo_str, cf.algo_str + '.npy')

    if mode == 'random':
        num_trajectories = int(
            cf.trajectory_resampling_factor * np.sqrt(average_free_space) *
            map_arr.size / (mean_path_length / cf.resolution))
    elif mode == 'transverse':
        num_trajectories = cf.trajectory_resampling_factor

    trajectories = \
        get_npy(file_name, calc_trajectories,
                map_arr, num_trajectories, mode=mode, resolution=cf.resolution,
                min_dist=cf.min_traj_length, max_dist=cf.max_traj_length,
                diagonal=cf.diagonal, verbose=True)
    if plot:
        plot_trajectories(trajectories, map_arr, map_res=cf.resolution)
        # save plot
        plot_file = re.sub('data', 'media', file_name)
        plot_file = re.sub('.npy', '.svg', plot_file)
        ensure_dir(plot_file)
        plt.show()
        plt.savefig(plot_file, format='svg')
        plt.close()

    return map_file, file_name


