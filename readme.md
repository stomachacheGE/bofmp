# Bayesian Occupancy Filter with Motion Pattern (BOFMP)

Welcome to my Master's thesis project BOFMP. This repo contains all the codes I used for running experiments for my thesis. However, due to limited time, those codes are just **workable** instead of **readable**. You may find it is painful to understand the codes, but I hope you can find something useful out of it. Tolerate and have fun :)

The whole project contains following prats:

  - Simulate human trajectories with A-star
  - Convert trajectories to motion probabilities used as network i/o
  - Training of neural networks
  - Tracking with BOFUM and BOFMP and visualization
  - Parameter tunning for BOFUM and BOFMP
  - Evaluate on test scenes with best parameters

First of all, go to root directory of this repo:
```sh
$ cd transition
```
Before you start anything, you should have a configuration file `config.py` in the root directory. Basically, this file contains all the parameters for training your network. Most important ones are

```python
# DATASET
project_folder = <project_foler> # this repo
data_folder =  project_folder + '/<data_folder_name>' # where the generated data goes to
map_folder = data_folder + '/maps' # put all the maps in this foler before generating data
training_maps = [<map_1>, <map_2>, ..., <map_n>] # maps used for trainning and validation (9:1)
test_maps = [<map_1>, <map_2>, ..., <map_n>] # maps used for testing

# SIMULATION
conditional_prob = True # caculate conditional probs or joint probs
diagonal = True # generate trajectories with diagonal directions or not

# TRAINING OUPUT
experiment_name = <experiment_name> # the trained model will live under ./trained_models/<experiment_name>
```

# Data generation

Once `<config_name>.py` is set up, run the following command from console to generate dataset used for trainnig network:

```sh
$ python data_generator/__init__.py -config <config_name>
```
Codes will use *multiprocessing* module from Python to generate data. This may take hours (~6h in my case). Then run following to put data from different maps together and it will show number of samples you have in your data:
```sh
$ python data_loader.py -config <config_name>
```
voila! You have data ready.

# Neural network trainning

Run the following to train your network:
```
$ THEANO_FLAGS='device=<gpu?>' python train.py -config <config_name>
```
Your trained model will be identified by `<experiment_name>` you specified in `<config_name>.py`.

# Tracking with BOFUM and BOFMP
Most of the codes for tracking lives under `./tracking`. We define a tracking case as a `scene`, which contains sensor data for each time step. You may want to check how tracking is working by running:

```sh
$ python tracking/scripts/visualize_tracking.py
```
However, to make it work properly, open that file and make sure you set all options correctly. For example, `cnn_model` should be the `<experiment_name>`. You can visualize the tracking process step by step (i.e., with *prediction* and *correction* steps) or only show for certain time steps or even with animation. When you visualize with animation, you can click on the cells on the map, and additional information will be shown. There are also a few shortcuts controlling pause/continue or updating tracking with new secene. Check that script for details.

# Parameters tuning
Before tuning parameter, open `tracking/scripts/parameter_tuning.py` to set neccessary options, e.g., where to get scene files, and parameter valude ranges. To find the best set of parameters for different filter setup, run
```sh
$ python tracking/scripts/parameter_tuning.py -cnn_model_name <cnn_model> -metric <metric>
```
`<cnn_model>` is the `<experiment_name>` for BOFMP, and simply type `bofum` for BOFUM. Optional arguments are:

- `-simulated_scenes`, bool, default False, whether the scenes are from simulation or real data
- `-blur_spatially`, bool, default False, whether use spatial blurring for BOFMP or not
- `-keep_motion`, bool, default False, whether use moving average velocity for BOFMP or not
- `-simulated_diagonal`, bool, default False, if `simulated_scenes=True`, this defines whether scenes have diagonal movements

Once the script finishes running, the best 5 sets of parameters will be printed on the console. The results of all the tries of parameters tuning will be saved in a folder under `./tracking/results`. In that folder, there is a file named as `summary.csv` that records every parameter set and their performance.

The best set of parameters that I tuned for **conditional** motion probabilities with **diagonal** movements can be retrieved by:

```python
from tracking.scripts.visualize_tracking import get_filter_kwargs
kwargs = get_filter_kwargs(<filter_name>, simulated_data=False, keep_motion=False, spatial_blur=False)
```
# Evaluation on test data
Once you have the best set of parameters, put them into `./tracking/scripts/evaluation_on_test_scenes.py` and also specify other variables, e.g., the file path of test scenes. Then run
```sh
$ python ./tracking/scripts/evaluation_on_test_scenes.py
```
and the evaluation results will prompt up once finishes running.

# Useful scripts and their functionalities
Except mentioned explictly, those scripts are under `./tracking/scripts/`.

| script | functionalities |
| ------ | ------ |
| `generate_simulation_scene.py` | genenerate simulated scenes from maps of real scenes |
| `generate_trajs_on_map_crop.py` | sample human trjectories and calculate motion probability on a small **map crop** |
| `generate_theme_plot.py` | generate the plot used in thesis that shows the basic idea of BOFMP |
| `best_prarameters_and_on_test_data.ipynb` | shows the best parameters used in thesis and evaluation results on test data|

There are also more ipython notebook under `./tracking/scripts/notebooks`, but there are still under development.

# Label real scene
There is also a small application that can be used for selecting and labeling real scenes with motion classes (e.g., turn/straight). Open `./tracking/scripts/selecting_scene.py` and change `button_names` to the classes you need, also specify `data_folder` where real scenes lives in. Run the script by
```sh
$ python ./tracking/scripts/selecting_scene.py
```
You can close the application anytime you want, and it will load previous result so that you can continue to label remaining scene files. The labeling result will be in a file `classification.pkl` under `data_folder`. Check `./tracking/scripts/save_labeled_scenes.ipynb` to see how to save classified scenes to a single file that can be used for tuning parameters or evaluation as test data.

**Have fun and happy coding.**