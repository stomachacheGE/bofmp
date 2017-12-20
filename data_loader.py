
import os
import imp
import argparse

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.occ_map_utils import load_map
from utils import ensure_dir

class Iterator(object):
    def __init__(self, inputs, labels, masks, batch_size):
        self.num_samples = len(inputs)
        self.batch_size = batch_size

        self.arrays = [np.array(array, dtype=np.float32) for array in
                       [inputs, labels, masks]]

        self.index = 0

    def next(self):
        # check if episode is done
        if self.get_n_samples() - self.index < self.batch_size:
            # reshuffle samples
            perm = np.random.permutation(self.num_samples)
            for array in self.arrays:
                array = array[perm]
            self.index = 0

        self.index += self.batch_size

        # get samples for current batch
        batch = [array[self.index - self.batch_size : self.index]
                 for array in self.arrays]

        # add singleton channel dimension
        batch_tmp = []
        for idx, array in enumerate(batch):
            #print("shape before is {}".format(str(array.shape)))
            if idx==0:
                # add channel axis for input ?
                array = np.expand_dims(array, 1)
            else:
                if array.ndim == 5:
                    # if last two dims are last vel and next vel,
                    # merge them into one dimension
                    old_shape = array.shape
                    array = np.reshape(array, (old_shape[0], old_shape[1], old_shape[2],
                                       old_shape[3]*old_shape[4]))
                array = np.transpose(array, (0, 3, 1, 2))
            #print("shape after is {}".format(str(array.shape)))
            batch_tmp.append(array)
        batch = batch_tmp

        return tuple(batch)

    def get_n_samples(self):
        return self.num_samples

    def get_n_batches(self):
        return np.floor(self.get_n_samples() / self.batch_size).astype(int)

def get_iters(config_path):

    cf = imp.load_source('config', config_path)
    ensure_dir(cf.training_data_path)

    inputs_filename = ['train_input', 'val_input', 'test_input']
    outputs_filename = ['train_output', 'val_output', 'test_output']
    masks_filename = ['train_mask', 'val_mask', 'test_mask']

    files = []
    for filenames in [inputs_filename, outputs_filename, masks_filename]:
        files_ = [cf.training_data_path+'/'+filename+'.npy'
                                          for filename in filenames ]
        files += files_

    # check if all files exist. If not,
    # delete existing files and re-generate.
    for f in files:
        if not os.path.isfile(f):
            for f in files:
                if os.path.isfile(f):
                    os.remove(f)
            i_o_m_files = [files[i:i+3] for i in range(0, 9, 3)]
            generate_datasets(*i_o_m_files, config_path=config_path)

    arrays = []
    for f in files:
        arrays.append(np.load(f))

    return [Iterator(arrays[i], arrays[i+3], arrays[i+6], batch_size=cf.batch_size) for i in range(3)]

def get_iters_map():
    # data_directory = '/home/doj2rng/dat/'
    data_directory = '/local/home/ful7rng/projects/occupancy/data'

    arrays = defaultdict(dict)
    for ix, data_set in enumerate(data_sets):
        input_str = data_directory + 'maps/' + maps[ix][0] + \
                    '/thresholded_20.png'
        arrays['inputs'][data_set], _, _ = load_map(input_str)

        label_str = data_directory + 'occ_maps/' + maps[ix][0] + '/' + \
                    method[ix] + '/' + os.path.split(method[ix])[-1] + '.npy'
        arrays['labels'][data_set] = np.load(label_str)

        mask_str = data_directory + 'occ_maps/' + method[ix] + '/' + \
                   maps[ix][0] + '_20.npy'
        arrays['masks'][data_set] = np.load(mask_str)

    iterators = [Iterator(arrays['inputs'][data_set][None, :, :],
                          arrays['labels'][data_set][None, :, :],
                          arrays['masks'][data_set][None, :, :], 1)
                 for data_set in data_sets]

    return iterators


def generate_datasets(inputs_files, outputs_files, masks_files, config_path):
    cf = imp.load_source('config', config_path)
    num_directions = cf.num_directions
    num_unique_vels = len(cf.unique_vel_idxs)
    train_data = []
    test_data = []
    for idx, maps in enumerate([cf.training_maps, cf.test_maps]):
        inputs = np.empty((0, cf.nn_input_size, cf.nn_input_size))
        if cf.conditional_prob:
            outputs = np.empty((0, cf.nn_output_size, cf.nn_output_size, num_directions, num_directions))
            masks = np.empty((0, cf.nn_output_size, cf.nn_output_size, num_directions, num_directions))
        else:
            outputs = np.empty((0, cf.nn_output_size, cf.nn_output_size, num_unique_vels))
            masks = np.empty((0, cf.nn_output_size, cf.nn_output_size, num_unique_vels))

        for map in maps:
            map_io_folder = cf.data_folder+'/network_io/'+map+'/'+cf.algo_str
            for data in ['inputs', 'outputs', 'masks']:
                f_name = map_io_folder + '/' + data + '.npy'
                if data == 'inputs':
                    #print("inputs shape:{}".format(str(np.array(inputs).shape)))
                    #print("file shape:{}".format(str(np.load(f_name).shape)))
                    inputs = np.concatenate([inputs, np.load(f_name)])
                elif data == 'outputs':
                    outputs = np.concatenate([outputs, np.load(f_name)])
                else:
                    # print("masks shape:{}".format(str(np.load(f_name).shape)))
                    masks = np.concatenate([masks, np.load(f_name)])
        if idx == 0:
            train_data.append(inputs)
            train_data.append(outputs)
            train_data.append(masks)
        else:
            test_data.append(inputs)
            test_data.append(outputs)
            test_data.append(masks)

    # split train_data into train/val data
    num_instances = train_data[0].shape[0]
    print("found {} instances.".format(num_instances))
    idxs = np.random.choice(num_instances, int(0.15 * num_instances), replace=False )
    left_idxs = [idx for idx in range(num_instances) if idx not in idxs]
    val_data = [data[idxs] for data in train_data]
    train_data_ = [data[left_idxs] for data in train_data]

    data_ = list(zip(train_data_, val_data, test_data))
    for i, filenames in enumerate([inputs_files, outputs_files, masks_files]):
        for j, filename in enumerate(filenames):
            ensure_dir(filename)
            np.save(filename, data_[i][j])

def get_map_crop(config_path, num=1, dataset='test', probs=False):
    cf = imp.load_source('config', config_path)
    ensure_dir(cf.training_data_path)

    filename = dataset + '_input'
    file = cf.training_data_path + '/' + filename + '.npy'
    inputs = np.load(file)

    if probs:
        filename = dataset + '_output'
        file = cf.training_data_path + '/' + filename + '.npy'
        outputs = np.load(file)

    print(inputs.shape)
    idxs = np.random.choice(np.arange(len(inputs)), num)
    maps = inputs[idxs].astype('bool')

    if not probs:
        return maps
    else:
        prob = outputs[idxs]
        return maps, prob

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

    train, val, test = get_iters(config_path)

    print("training data: {}".format(train.get_n_samples()))
    print("validation data: {}".format(val.get_n_samples()))
    print("test data: {}".format(test.get_n_samples()))
    train.next()















