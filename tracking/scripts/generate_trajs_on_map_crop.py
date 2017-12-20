import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from utils.scene_utils import get_scenes
from data_generator.astar_ped_sim.astar_traj_generator import _sample_trajectories
from data_generator.human_mcm import Grid_HMM
from utils import nd_gaussian
from utils.occ_map_utils import greens_cm
from tracking.animation import Plot
from utils.plot_utils import plot_4d_tensor


def get_cost_map(map_):
    collision_cost = 2000

    blur1_width = 5
    blurred_map1 = cv2.GaussianBlur(map_.astype(float),
                                    (blur1_width, blur1_width), 0)
    blur2_width = 11
    blurred_map2 = cv2.GaussianBlur(map_.astype(float),
                                    (blur2_width, blur2_width), 0)
    blur1_cost = 2
    blur2_cost = 1
    cost_map = collision_cost * map_.astype(float) + \
               blur1_cost * blurred_map1 + blur2_cost * blurred_map2
    return cost_map


def sample_trajs(padded_map, map_, w, h, num_trajectories, diagonal=False, mode='random', verbose=True):
    trajectories = []

    cost_map = get_cost_map(padded_map)

    start_locations = np.array(np.where(np.logical_not(map_))).T
    start_locations += np.array([w // 2, h // 2])

    if mode == 'traverse':
        num_locations = len(start_locations)
        for idx, start in enumerate(start_locations):
            if verbose and idx % 1 == 0:
                print('sampling trajectories for possible start locations: {}/{}'.format(
                    idx + 1, num_locations))
                x, y = start
                #                 center_map = np.ones_like(padded_map)
                #                 center_map[x-w//2:x+w//2, y-h//2:y+h//2] = padded_map[x-w//2:x+w//2, y-h//2:y+h//2]
                center_map = padded_map
                trajectories_ = _sample_trajectories(start, center_map, cost_map,
                                                     num_trajectories, diagonal=diagonal)
                if trajectories_ is not None:
                    trajectories += trajectories_
    elif mode == 'random':
        for ix in range(int(num_trajectories)):
            if verbose and ix % 10 == 0:
                print('sampling trajectories: {}/{}'.format(
                    ix + 1, num_trajectories))
            # loop to find valid path
            while True:
                start = random.sample(start_locations, 1)[0]
                x, y = start
                center_map = padded_map
                #                 center_map = np.ones_like(padded_map)
                #                 center_map[x-w//2:x+w//2, y-h//2:y+h//2] = padded_map[x-w//2:x+w//2, y-h//2:y+h//2]
                print("start is :{}".format(start))
                trajectory = _sample_trajectories(start, center_map, cost_map, 1, diagonal=diagonal)
                if trajectory is not None and len(trajectory) > 0:
                    break
            trajectories += trajectory

    return trajectories





def get_transition_map(trajs, padded_map, w, h, diagonal=False, conditional=True):
    mcm = Grid_HMM(np.array(padded_map.shape).astype(int))
    for idx, trajectory in enumerate(trajs):
        if idx % 100 == 0:
            print("Add transitions of {}/{} trajectory to the mcm.".format(idx + 1, len(trajs)))
        for t in range(trajectory.shape[0] - 2):
            from_ = trajectory[t, :]
            current = trajectory[t + 1, :]
            to = trajectory[t + 2, :]
            mcm.add_transition(from_, current, to)

    transition_probs = mcm.get_transition_probs(conditional, diagonal)
    return transition_probs[w // 2:w + w // 2, h // 2:h + h // 2]

def rescale_trajectory(trajectory, resolution):
    # scale path to real world extent
    return np.array(trajectory + np.array([0.5] * 2)[None, :]) * resolution

def onclick_traj_random(event, fig, plot, fig_1, raw_trajs, probs, **kwargs):
    try:
        for i in range(len(plot.lines)):
            plot.axes.lines.remove(plot.lines[i])
    except AttributeError, ValueError:
        pass

    ix, iy = event.xdata, event.ydata
    coords = np.floor(np.array([ix, iy]) / 0.2).astype(int)
    print(coords)

    clicked = np.zeros_like(map_)
    x, y = coords[0], coords[1]
    clicked[x, y] = 1
    plot.set_axes_data("occupancy_axes", clicked)

    g_len = 300
    gaussian_res = float(g_len) / 32
    center = [round((x + .5) * gaussian_res), round((y + .5) * gaussian_res)]
    gaussian = nd_gaussian([g_len, g_len], center, cov=14)
    plot.set_axes_data("gaussian_axes", gaussian)

    w, h = kwargs.get('w', 0) // 2, kwargs.get('h', 0) // 2
    x_padded, y_padded = x + w, y + h
    trajs_here = []
    for idx, traj in enumerate(raw_trajs):
        if [x_padded, y_padded] in traj.tolist():
            traj_ = raw_trajs[idx] - np.array([w, h])
            trajs_here.append(rescale_trajectory(traj_, .2))
    print("Found %d trajs going through here" % len(trajs_here))
    idx = np.random.randint(len(trajs_here))
    print("Take random traj at idx %d" % idx)
    idx = 3
    colors = cm.Dark2(np.linspace(0, 1, len(trajs_here)))
    plot.lines = [plot.axes.add_line(Line2D([], [], zorder=13, color=colors[idx]))]

    xs, ys = trajs_here[idx].T[0], trajs_here[idx].T[1]
    #         if idx <= 5:
    #             print(trajs_here[idx])
    plot.lines[0].set_data(xs, ys)

    fig.canvas.draw()

    fig_1.clear()
    plot_4d_tensor(probs[x, y], fig=fig_1)
    fig_1.canvas.draw()

if __name__ == '__main__':

    fname = '/home/ful7rng/projects/transition/tracking/data/real_scenes/selected_scenes_for_tuning_param_from_100_11.npy'
    sample_rate = 3
    scenes = get_scenes(random_file=False,
                      min_time_interval=3.5,
                      max_time_interval=1e6,
                      sample_rate=sample_rate,
                      file_name=fname)[0]
    idx = 514
    scene = scenes[idx]
    map_ = scene.static_map
    w, h = map_.shape
    padded_map = np.pad(map_, ((w // 2, w // 2), (h // 2, h // 2)), mode='edge')

    raw_trajs = sample_trajs(padded_map, map_, w, h, 4, diagonal=True, mode='traverse', verbose=True)
    transition_probs = get_transition_map(raw_trajs, padded_map, w, h, diagonal=True)

    fig = plt.figure(figsize=(8, 8))
    map_axes = fig.add_subplot(111)
    plot = Plot(map_axes, map_, .2, title='')

    plot.add_custom_image("gaussian_axes", image=np.zeros((300, 300)),
                          cmap=greens_cm, extent=(0, 6.4, 0., 6.4),
                          zorder=-1)
    plot.axes.set_xlabel('m')
    plot.axes.set_ylabel('m')
    fig_1 = plt.figure(figsize=(5, 5))
    fig.canvas.mpl_connect('button_press_event',
                           lambda event: onclick_traj_random(event, fig, plot, fig_1,
                                                             raw_trajs, transition_probs,
                                                             w=w, h=h))
    plt.show()

