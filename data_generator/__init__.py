import imp
import os
import threading
import argparse

import numpy as np

from ped_sim import generate_trajectories
from transistional_map import  generate_transitional_map
from map2nnio import  generate_network_io
from multiprocessing import Pool

def generate_data(map_str):
    map_path, traj_path = generate_trajectories(map_str, config_path)
    trans_map_path, trans_count_path = generate_transitional_map(map_path, traj_path, config_path)
    generate_network_io(trans_map_path, trans_count_path, map_path, config_path)

if __name__ == "__main__":

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

    # get a list of maps
    map_list = cf.training_maps + cf.test_maps
    np.random.shuffle(map_list)
    p = Pool(10)
    p.map_async(generate_data, map_list)
    p.close()
    p.join()
