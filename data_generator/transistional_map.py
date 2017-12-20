
import os
import fnmatch
import re
import imp

import numpy as np
import matplotlib.pyplot as plt
import cv2

from human_mcm import Grid_HMM
from utils import get_npy, blur2int, ensure_dir
from utils.occ_map_utils import load_map, display_occ_map,  display_trans_map


def generate_transitional_map(map_path, traj_path, config_path, plot=False):
    cf = imp.load_source('config', config_path)
    map_, _, _ = load_map(map_path)

    trans_map_path = re.sub('trajectories', 'transition_maps', traj_path)
    trans_counts_path = re.sub('trajectories', 'transition_counts', traj_path)

    transition_probs, transition_counts = \
        get_npy([trans_map_path, trans_counts_path], calc_transitional_map, map_path,
                traj_path, config_path)

    if not os.path.isfile(trans_map_path):
        ensure_dir(trans_map_path)
        np.save(trans_map_path, transition_probs)

    if not os.path.isfile(trans_counts_path):
        ensure_dir(trans_counts_path)
        np.save(trans_counts_path, transition_counts)

    if plot:
        fig = plt.figure()
        for last in range(4):
            for next in range(4):
                order = last * 4 + next
                display_trans_map(transition_probs, map_, last, next, order,
                                  cost_map_res=cf.resolution)
        fig.subplots_adjust(left=0.05, bottom=0.06, right=1, top=0.96, wspace=0, hspace=0.47)
        plt.show()

    return trans_map_path, trans_counts_path


def calc_transitional_map(map_path, traj_path, config_path):
    cf = imp.load_source('config', config_path)

    map_, _, _ = load_map(map_path)
    mcm = Grid_HMM(np.array(map_.shape).astype(int))
    trajectories = np.load(traj_path)
    for idx, trajectory in enumerate(trajectories):
        if idx % 100 == 0:
            print("Add transitions of {}/{} trajectory to the mcm.".format(idx+1, len(trajectories)))
        trajectory = np.round((trajectory / cf.resolution) - 0.5).astype(int)
        #print(str(trajectory))
        for t in range(trajectory.shape[0] - 2):

            from_ = trajectory[t, :]
            current = trajectory[t + 1, :]
            to = trajectory[t + 2, :]
            mcm.add_transition(from_, current, to)

    conditional = cf.conditional_prob
    transition_probs = mcm.get_transition_probs(conditional, cf.diagonal)

    return transition_probs, mcm.get_transition_counts()


if __name__ == '__main__':
    tags = ['astar_cost_sg_random_6_20', 'real']

    # traj_folder = os.path.expanduser('~') + '/dat/trajectories/'
    traj_folder = '/local/home/ful7rng/projects/occupancy/data/trajectories/'

    # file_list = [file for file in glob(
    #     traj_folder + '**/*.npy', recursive=True)]
    # file_list = [file for file in glob(
    #     traj_folder + '**/*.npy')]
    file_list = []
    for root, dirnames, filenames in os.walk(traj_folder):
        for filename in fnmatch.filter(filenames, '*.npy'):
            file_list.append(os.path.join(root, filename))
    print(file_list)

    # filter out files containing tag
    file_list = [file for file in file_list for tag in tags if tag in file]
    # print(file_list)

    np.random.shuffle(file_list)
    for traj_file in file_list:
        print(traj_file)
        tags = ['/'] + traj_file.split('/')
        home_lvl = tags.index('home') + 4

        map_folder = re.sub('trajectories', 'maps', os.path.join(
            *tags[:home_lvl + 3]))
        map_, map_origin, resolution = load_map(
            os.path.join(map_folder, 'thresholded_20.png'))

        trajectories = np.load(traj_file, encoding='latin1')

        occ_map_file = re.sub(
            'trajectories', 'occ_maps', os.path.join(
                                         os.path.join(*tags[:home_lvl + 3]),
                                         'from_traj',
                                         os.path.join(*tags[home_lvl + 3:])))

        if 'real' in tags:
            blur = 0.1
        else:
            blur = 0.25


        occ_map = \
            get_npy(occ_map_file, calc_occ_map, map_,
                    trajectories, resolution=resolution,
                    map_origin=map_origin, blur=blur,
                    verbose=True)

        # save total occupancy map
        occ_map_folder = re.sub('trajectories', 'occ_maps', os.path.join(
            os.path.join(*tags[:home_lvl + 3]), 'from_traj', tags[home_lvl + 3]))
        tot_occ_map_file = os.path.join\
            (occ_map_folder, tags[home_lvl + 3] + '.npy')
        tot_occ_map = get_npy(
            tot_occ_map_file, np.zeros_like, map_, dtype=float)
        tot_occ_map += occ_map

        ensure_dir(tot_occ_map_file)
        np.save(tot_occ_map_file, tot_occ_map)

        # save plot of total occupancy map
        display_occ_map(tot_occ_map, map_, cost_map_res=resolution,
                        cost_map_origin=map_origin)
        tot_plot_file = re.sub('dat', 'media',
                               re.sub('.npy', '.svg', tot_occ_map_file))
        ensure_dir(tot_plot_file)
        plt.savefig(tot_plot_file, format='svg')
        plt.close()

