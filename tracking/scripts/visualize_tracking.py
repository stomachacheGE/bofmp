


import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from utils.occ_map_utils import load_map
from utils.scene_utils import animate_scenes, get_scenes
from data_loader import get_map_crop
from tracking.filters import BOFUMRealdata, naiveBOFUM, conditionalBOFUM
from tracking.visualize import VisualizeSimulation, VisualizeRealdata
from tracking.animation import TrackingAnimSimulation, TrackingAnimRealdata

def on_press(event, animation):
    global scenes
    if event.key == 'n':
        idx = np.random.choice(np.arange(len(scenes)))
        print("Refresh animation with scene at idx {}".format(idx))
        scene = scenes[idx]
        animation.update(scene)

def map_portfolio(name):

    def propose_x_y(map):
        locations = np.array(np.where(1-map)).T.tolist()
        idx = np.random.randint(0, len(locations), 1)[0]
        x, y = locations[idx]
        return x, y

    def make_T_section(size):
        map_size = size
        map = np.ones((map_size, map_size), dtype=bool)
        lane_length = size // 2

        for j in range(map_size / 2):
            map[j + 1:j + lane_length + 1, j] = 0
        for j in range(map_size / 2, map_size):
            map[map_size - j:map_size - j + lane_length, j] = 0
        for j in range(map_size / 2 - lane_length / 2, map_size / 2 + lane_length / 2 + 1):
            map[map_size / 2:map_size, j] = 0
        x, y = map_size - map_size / 4, map_size / 2
        return map, x, y

    if name == 'minimum':
        map = np.array([[1, 0, 0, 0, 0, 0, 1]] * 20, dtype=bool)
        x, y = propose_x_y(map)

    elif name == 'T-section':
        map, x, y = make_T_section(31)

    elif name == 'empty_map':
        map_size = 62
        map = np.zeros((map_size, map_size), dtype=bool)
        x, y = int(map_size/4*3), int(map_size/2)

    elif name == 'big_T_section':
        map, x, y = make_T_section(91)

    elif name == 'simple_map':
        # load simple map
        map_path = '/local/home/ful7rng/projects/transition/data/maps/simple_map/thresholded_20.png'
        map, _, _= load_map(map_path)
        x, y = propose_x_y(map)

    elif name == 'any':
        config_path = '/local/home/ful7rng/projects/transition/config.py'
        map = get_map_crop(config_path, num=1)[0]
        x, y = propose_x_y(map)

    width, height = map.shape
    return map, width, height, x, y


def get_filter_kwargs(filter, simulated_data=False, keep_motion=False, spatial_blur=False):

    if filter == 'BOFUM':

        if simulated_data:
            kwargs = dict(omega=0.0164534434234,
                          extent=7,
                          noise_var=0.555536471101,
                          measurement_lost=8,
                          name='BOFUM',
                          verbose=False)

        else:
            kwargs = dict(omega=0.152265835237,
                          extent=5,
                          noise_var=0.676892668163,
                          measurement_lost=8,
                          name='BOFUM',
                          verbose=False)

    elif filter == 'BOFMP':

        if simulated_data:
            kwargs = dict(omega=0.0353639265597,
                          extent=7,
                          noise_var=0.372650880654,
                          measurement_lost=8,
                          name='BOFMP',
                          verbose=False)
        else:

            if keep_motion and not spatial_blur:

                kwargs = {
                    'omega': 0.0263632138087,
                    'extent': 7,
                    'noise_var': 0.743921945138,
                    'name': 'BOFMP with motion keeping',
                    'measurement_lost': 8,
                    'keep_motion': True,
                    'window_size': 4,
                    'initial_motion_factor': 0.562946012048,
                    'keep_motion_factor': 0.706594184009
                }

            elif spatial_blur and not keep_motion:
                kwargs = {  'blur_extent' : 5,
                            'blur_var': 1.09280355491,
                            'noise_var': 0.635747744503,
                            'extent': 5,
                            'omega': 0.0996668473785,
                            'name': 'BOFMP with spatial blurring'
                }

            elif not spatial_blur and not keep_motion:

                kwargs = {
                    'omega': 0.191234382275,
                    'extent': 5,
                    'noise_var': 0.644828151836,
                    'name': 'BOFMP'
                }

            else:
                raise ValueError("Cannot get required keyword arguments since they are not available.")
    else:
        raise ValueError("Filter {} does not exist.".format(filter))


    flag = "simulated data" if simulated_data else "real data"
    print("Get keyword arguments for {} on {}".format(kwargs['name'], flag))
    print(kwargs)

    return kwargs


if __name__ == "__main__":

    simulated_data = False
    filter_on_simulated_data = True
    animation = True
    cnn_model = '04_DIAGONAL_TRUE_CONDITIONAL_TRUE'
    scene_folder = '/local/data/scenes/100_11'

    if simulated_data:
        map_name = 'T-section'
        force_predict = False
        if map_name == 'any':
            force_predict = True
        map_, width, height, x, y = map_portfolio(map_name)
        num_steps = 15

    else:
        laser_frequency = 12
        min_time_interval = 4
        max_time_interval = 1e6
        # scenes for test, suitable for showing in thesis: [27]
        fname = '/home/ful7rng/projects/transition/tracking/data/simulated_scenes/500_simulated_scenes_diagonal_for_testing_from_80_new.npy'
        # for showing idea of moving average : [200]
        #fname = '/home/ful7rng/projects/transition/data/244_scenes_for_test_from_80_new.npy'
        scenes = get_scenes(data_folder=scene_folder, random_file=True,
                          min_time_interval=min_time_interval,
                          max_time_interval=max_time_interval,
                          file_name=None,
                          simulated_scenes=False)[0]
        idx = np.random.randint(len(scenes))
        #idx = 27
        scene = scenes[idx]
        animate_scenes([scene])
        animate_scenes([BOFUMRealdata.scene_preprocessing(copy(scene))])
        map_ = scene.static_map
        sample_rate = 3
        num_steps = int(laser_frequency * min_time_interval / sample_rate)
        force_predict = False

    filters = []

    ###########################################
    ############### BOFUM #####################
    ###########################################
    bofum_kwargs = get_filter_kwargs('BOFUM', simulated_data=filter_on_simulated_data)
    bofum = naiveBOFUM(map_, simulated_data=simulated_data, **bofum_kwargs)
    filters.append(bofum)

    ###########################################
    ############### BOFMP #####################
    ###########################################
    bofmp_kwargs = get_filter_kwargs('BOFMP', simulated_data=filter_on_simulated_data)
    bofmp = conditionalBOFUM(map_, cnn_model,
                             simulated_data=simulated_data,
                             force_predict=force_predict,
                              **bofmp_kwargs)
    filters.append(bofmp)




    ###########################################
    ###### Visualize as stactic plots  ########
    ###########################################
    if not animation:


        if simulated_data:
            visualization = VisualizeSimulation(filters, num_steps, num_targets=1,
                                                  dynamically=False,
                                                  tracking=True,
                                                  show_map=True,
                                                  num_col=3,
                                                  traj_overlay=True,
                                                  show_colorbar=True,
                                                  diagonal=True,
                                                  show_metric=False,
                                                  # init_locs=[[20, 15]],
                                                  # init_moves=['LEFT'],
                                                  show_elements=['pred', 'corr', 'obser']
                                      )

        else:
            visualization = VisualizeRealdata(filters, num_steps, scene,
                                                  dynamically=False,
                                                  show_map=True,
                                                  show_seen=True,
                                                  num_col=4,
                                                  traj_overlay=True,
                                                  show_colorbar=False,
                                                  show_metric=True,
                                                  tracking=True,
                                                  show_at=[8, 10, 13, 16]
                                      )

        visualization.start()

    ###########################################
    ######## Visualize as animation  ##########
    ###########################################
    else:

        ########################### Shortcuts for animation ##############################
        # To use shortcuts, first click on tracking figure since shortcuts are only
        # linked to this figure.
        # 1. You may click any cell on map for any filter. According to accessories defined
        #    below, you may see the motion pattern probabilities or velocities on that cell.
        # 2. Pressing "Space" : pause or start animation
        # 3. Pressing "n": refresh animation with a new scene randomly take from global
        #    variable scenes
        ###################################################################################

        accessories = ['motion_pattern', 'velocities']
        if simulated_data:
            ani = TrackingAnimSimulation(filters, num_steps, num_targets=3,
                                           plot_map=True,
                                           diagonal=False,
                                           accessories=accessories
                                           )
        else:
            ani = TrackingAnimRealdata(filters, num_steps, scene,
                                        plot_map=True,
                                        plot_seen=True,
                                        accessories=accessories)

        ani.fig.canvas.mpl_connect('key_press_event', lambda event: on_press(event, ani))
        #ani.save('/home/ful7rng/Desktop/tracking_with_moving_average.mp4', writer='ffmpeg')
        plt.show()
