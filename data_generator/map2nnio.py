import os
import imp
import re
import random

import numpy as np

from utils import ensure_dir
from utils.occ_map_utils import load_map, free_space
from human_mcm import Grid_HMM


def generate_network_io(trans_map_path, trans_counts_path, map_path, config_path):

    network_io_dir = '/'.join(trans_map_path.split('/')[:-1])
    network_io_dir = re.sub('transition_maps', 'network_io', network_io_dir)
    print(network_io_dir)
    ensure_dir(network_io_dir)

    paths = []
    for data_name in ["inputs", "outputs", "masks"]:
        path = os.path.join(network_io_dir, data_name + ".npy")
        ensure_dir(path)
        paths.append(path)

    # inputs_path = os.path.join(network_io_dir, "inputs.npy")
    # outputs_path = os.path.join(network_io_dir, "outputs.npy")
    # ensure_dir(inputs_path)
    # ensure_dir(outputs_path)

    all_path_exist = np.all([os.path.isfile(path) for path in paths])

    if all_path_exist:
        print("Network input/outputs already exsist in {}".format(network_io_dir))
    else:
        # if os.path.isfile(inputs_path):
        #     os.remove(inputs_path)
        # if os.path.isfile(outputs_path):
        #     os.remove(outputs_path)
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)
        samples = sample_network_io(trans_map_path, trans_counts_path, map_path, config_path)
        path_and_data = dict(zip(paths, samples))
        for path in path_and_data:
            np.save(path, path_and_data[path])

def flatten_fn(velocities, conditional, vel_idxs):
    """ Flatten matrix of dimension [width, height, num_x_velocities, num_y_velocities,
     num_x_velocities, num_y_velocities] to:
      1. [width, height, num_movements, num_movements], if network models conditional probs.
      2. [width, height, num_vel_idxs], if network models joint probs. """

    def flatten_to_4d(transition_probs):
        width, height = transition_probs.shape[0], transition_probs.shape[1]
        num_vel = len(velocities)
        res = np.zeros((width, height, num_vel, num_vel), dtype=transition_probs.dtype)
        for i, vel_ls in enumerate(velocities):
            for j, vel_nx in enumerate(velocities):
                idx_ls_mv = Grid_HMM.two_d_vel_to_idx(vel_ls)
                idx_nx_mv = Grid_HMM.two_d_vel_to_idx(vel_nx)
                res[:, :, i, j] = transition_probs[:, :,
                                  idx_ls_mv[0], idx_ls_mv[1],
                                  idx_nx_mv[0], idx_nx_mv[1]]
        return res

    def flatten_to_3d(transition_probs):
        width, height = transition_probs.shape[0], transition_probs.shape[1]
        num_vel = len(vel_idxs)
        res = np.zeros((width, height, num_vel), dtype=transition_probs.dtype)
        for i, vel_idx in enumerate(vel_idxs):
            # print(vel_idx)
            # since joint probs of unique vels sum to 0.5,
            # multiplying it with 2 make it a probability
            res[:, :, i] = 2 * transition_probs[:, :,
                                            vel_idx[0], vel_idx[1],
                                            vel_idx[2], vel_idx[3]]
        return res

    if conditional:
        return flatten_to_4d
    else:
        return flatten_to_3d


def sample_network_io(trans_map_path, trans_counts_path, map_path, config_path):

    cf = imp.load_source('config', config_path)

    flatten = flatten_fn(cf.velocities, cf.conditional_prob, cf.unique_vel_idxs)

    input_size = cf.nn_input_size
    output_size = cf.nn_output_size

    map_arr_, _, _ = load_map(map_path)
    trans_probs = np.load(trans_map_path)
    trans_counts = np.load(trans_counts_path)

    average_free_space = np.mean(free_space(map_arr_))
    # num_samples = int(cf.nn_io_resampling_factor * average_free_space * \
    #                   map_arr_.size / cf.nn_output_size ** 2)
    num_samples = int(cf.nn_io_resampling_factor * \
                      map_arr_.size / cf.nn_output_size ** 2)

    inputs, outputs, masks = [], [], []

    for sample_ix in range(num_samples):
        if sample_ix % 10 == 0:
            print('generating network samples: {}/{}'.format(sample_ix + 1, num_samples))
        while True:
            # find a map segment which is suitable for training
            left, top = [random.randint(0, map_arr_.shape[ix] - input_size) for ix
                         in [0, 1]]
            x_in, y_in = [slice(start, start + input_size) for start in [left, top]]
            x_out, y_out = [slice(start + int((input_size - output_size) / 2),
                                 start + int((input_size + output_size) / 2))
                            for start in [left, top]]

            input = map_arr_[x_in, y_in].copy()
            output = trans_probs[x_out, y_out].copy()
            trans_counts_patch = trans_counts[x_out, y_out].copy()
            if cf.conditional_prob:
                axis = (4, 5)
            else:
                axis = (2, 3, 4, 5)
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = np.sum(trans_counts_patch, axis=axis, keepdims=True) / np.sum(trans_counts_patch)
                mask[~np.isfinite(mask)] = 0
            for ax in axis:
                mask = np.repeat(mask, trans_counts.shape[ax], axis=ax)

            if np.mean(map_arr_[x_out, y_out]) >= 0.5 or np.mean(input) > 0.5:
                # not enough empty space to predict
                continue

            break

        # inputs.append(input.copy())
        # outputs.append(flatten(output).copy())
        # masks.append(flatten(mask).copy())



        # augment that segment 8 times
        # for mirror in [False, True]:
        #     input_ = input
        #     output_ = output
        #     mask_ = mask
        #     if mirror:
        #         input_ = np.fliplr(input)
        #         output_ = flip(output)
        #         mask_ = flip(mask)
        #     for num_rot in range(4):
        #         input__ = np.rot90(input_, k=num_rot)
        #         output__ = np.rot90(output_, k=num_rot)
        #         mask__ = np.rot90(mask_, k=num_rot)
        #         inputs.append(input__.copy())
        #         outputs.append(output__.copy())
        #         masks.append(mask__.copy())

        # augment that segment 8 times
        for mirror in [False, True]:
            input_ = input
            output_ = output
            mask_ = mask
            if mirror:
                # flip left/right visually means
                # filp up/down the numpy array
                input_ = np.flipud(input)
                output_ = flip(output)
                mask_ = flip(mask)
            for num_rot in range(4):
                input__ = np.rot90(input_, k=num_rot)
                output__ = rotate(output_, k=num_rot)
                mask__ = rotate_masks(mask_, k=num_rot)
                inputs.append(input__.copy())
                outputs.append(flatten(output__).copy())
                masks.append(flatten(mask__).copy())

    return inputs, outputs, masks


def flip(probs):
    # spatial dimensions of numpy array flip updown
    probs = np.flip(probs, axis=0)
    # velocity directions of left/right change to right/left
    probs = np.flip(probs, axis=2)
    probs = np.flip(probs, axis=4)
    return probs


def flip_masks(masks):
    # spatial dimensions of numpy array flip updown
    masks = np.flip(masks, axis=0)
    # velocity directions of left/right change to right/left
    masks = np.flip(masks, axis=2)
    return masks


def rotate(probs, k):
    """ Rotate the probs 90 degrees counter-clockwise k times.

    Note that directions of probabilities change after rotations. For example, if rotate once,
    the probability of P{left|down} changes to P{down|right}.
    """
    if k == 0:
        return probs

    # spatial dimensions rotate counter-clockwise
    probs = np.rot90(probs, k)
    # velocitiy dimensions rotate counter-clockwise
    probs = np.rot90(probs, k, (2, 3))
    probs = np.rot90(probs, k, (4, 5))
    return probs


def rotate_masks(masks, k):
    """ Rotate the masks 90 degrees counter-clockwise k times.

    """
    if k == 0:
        return masks

    masks = np.rot90(masks, k=k)
    masks = np.rot90(masks, k, (2, 3))
    return masks
