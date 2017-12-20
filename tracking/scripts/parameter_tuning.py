import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import argparse

from tracking.filters import conditionalBOFUM, naiveBOFUM
from tracking.tracking_param_tuning import parameter_tuning


if __name__ == '__main__':

    def str2bool(string):
        if string == 'True':
            return True
        elif string == 'False':
            return False

    parser = argparse.ArgumentParser()
    parser.add_argument('-cnn_model_name')
    parser.add_argument('-metric')
    parser.add_argument('-simulated_scenes', type=str2bool, nargs='?', default=False)
    parser.add_argument('-simulated_diagonal', type=str2bool,  nargs='?', default=False)
    parser.add_argument('-blur_spatially', type=str2bool, nargs='?', default=False)
    parser.add_argument('-keep_motion', type=str2bool, nargs='?', default=False)
    parser.add_argument('-keep_naive_bofum', type=str2bool, nargs='?', default=False)



    args = vars(parser.parse_args())
    print(args)

    cnn_model = args['cnn_model_name']

    if cnn_model == 'bofum':
        bofum = naiveBOFUM
        model_options = {}
    else:
        bofum = conditionalBOFUM
        model_options = {'blur_spatially': args['blur_spatially'],
                         'keep_motion': args['keep_motion'],
                         'keep_naive_bofum': args['keep_naive_bofum']}


    measurement_lost = 8
    num_scenes = 500
    num_tries = 100

    simulated_scenes = args['simulated_scenes']
    simulated_scenes_diagonal = args['simulated_diagonal']
    if simulated_scenes:
        if simulated_scenes_diagonal:
            scene_file = '/home/ful7rng/projects/transition/propagation/scenes/scenes_for_tuning_params.npy'
        else:
            scene_file = '/home/ful7rng/projects/transition/propagation/scenes/scenes_not_diagonal_for_tuning_params.npy'
    else:
        scene_file = '/home/ful7rng/projects/transition/data/selected_scenes_from_100_11.npy'
    parameter_tuning(cnn_model, args['metric'], model_options, measurement_lost, num_scenes, simulated_scenes,
                     simulated_scenes_diagonal,
                     bofum=bofum,
                     min_time_interval=4,
                     num_tries=num_tries, num_subprocess=10,
                     scene_file=scene_file)