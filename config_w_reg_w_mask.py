import os
import imp

from metrics import categorical_crossentropy_void
from lasagne.updates import adam

# DATASET
project_folder = '/local/home/ful7rng/projects/transition'
data_folder =  project_folder + '/data'
map_folder = data_folder + '/maps'
training_maps = ['intel_lab', 'office2', 'office1', '79', 'belgioioso', '100_7', 'mit_csail']
test_maps = ['100_11']

# SIMULATION
trajectory_sampling_mode = 'transverse'
resolution = 0.2  # map resolution, meter per pixel
diagonal = False
trajectory_resampling_factor = 5 # average number of trajectories per pixel
min_traj_length =  6# in meters
max_traj_length = 20  # in meters
algo_str = 'astar_cost_' + trajectory_sampling_mode + '_' + \
           str(min_traj_length) + '_' + \
           str(max_traj_length) + '_' + \
           str(trajectory_resampling_factor)


# NETWORK
nn_input_size = 32
nn_output_size = 32
nn_io_resampling_factor = 10

# TRAINING
training_data_path = data_folder+'/io_for_training/'+algo_str
seed = 0
learning_rate = 0.000151125753584
lr_sched_decay = 0.998524022851
num_epochs = 100
# target_lr_rate = 1e-6
# lr_sched_decay = (target_lr_rate / learning_rate) ** (1.0 / num_epochs) # Applied each
# epoch
weight_decay = 4.57573361749e-05
# weight_decay = 0
max_patience = 15
output_norm_fac = 1
loss_function = categorical_crossentropy_void
optimizer = adam
batch_size = 128
no_mask = False

# TRAINING OUPUT
experiment_name = '20_ALL_MAPS'
savepath = project_folder + '/trained_models/' + experiment_name

# Architecture
n_filters_first_conv = 8
n_pool = 2
growth_rate = 12
n_layers_per_block = 5
dropout_p = 0.349153524121

model_path = project_folder + '/architectures/fc_dense_net.py'
net = imp.load_source('Net', model_path).FCDenseNet(
    input_shape=(None, 1, None, None),
    n_directions = 4,
    n_filters_first_conv=n_filters_first_conv,
    n_pool=n_pool,
    growth_rate=growth_rate,
    n_layers_per_block=n_layers_per_block,
    dropout_p=dropout_p)