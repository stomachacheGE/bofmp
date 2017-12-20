
import os
import datetime
import random
import signal

import time

import numpy as np
import matplotlib.pyplot as plt
import cv2

from .a_star_grid import AStarGrid
from data_generator.multi_agent_planning.scripts.pyastar import astar_path
from utils import ensure_dir
from utils.occ_map_utils import free_space, show_map


def sample_sg_area(map_path_):
    goals_path = map_path_ + 'goals_20/'
    goal_files = os.listdir(goals_path)
    # get random goal file
    goal_file = random.sample(goal_files, 1)
    goal_arr = plt.imread(goals_path + goal_file[0])
    # filter out red part
    goal_arr = np.logical_and(goal_arr[..., 0] > 0, goal_arr[..., 1] == 0)
    # get list of pixels in that area
    goals = np.array(np.where(goal_arr)).T.tolist()
    # sample random pixel
    goal = random.sample(goals, 1)
    return np.array((goal[0][::-1]))


def sample_sg_random(goal_arr):
    # get list of pixels in that area
    goals = np.array(np.where(free_space(goal_arr))).T.tolist()
    # sample random pixel
    if goals:
        goal = random.sample(goals, 1)
        return np.array((goal[0]))
    # in case there is no possible sample result
    else:
        return None

def sample_trajectories(map_arr, num_trajectories, mode='transverse',
                        min_dist=10, max_dist=20, min_traj_len=None, verbose=False,
                        blur1_cost=2, blur2_cost=1, diagonal=True):

    # calculate cost map: make walls expensive and blur them
    collision_cost = 2000

    blur1_width = 5
    blurred_map1 = cv2.GaussianBlur(map_arr.astype(float),
                                   (blur1_width, blur1_width), 0)
    blur2_width = 11
    blurred_map2 = cv2.GaussianBlur(map_arr.astype(float),
                                   (blur2_width, blur2_width), 0)

    cost_map = collision_cost * map_arr.astype(float) + \
               blur1_cost * blurred_map1 + blur2_cost * blurred_map2
    trajectories = []
    if mode=='random':
        for ix in range(int(num_trajectories)):
            if verbose and ix % 10 == 0:
                print('sampling trajectories: {}/{}'.format(
                    ix + 1, num_trajectories))
            # loop to find valid path
            while True:
                # loop over tries for one start pixel
                start = sample_sg_random(map_arr)
                #print("start is :{}".format(start))
                trajectory = _sample_trajectories(start, map_arr, cost_map, 1,
                                  min_dist, max_dist, min_traj_len, diagonal)
                if trajectory is not None and len(trajectory) > 0:
                    break
            trajectories += trajectory
    elif mode=='transverse':
        start_locations = list(zip(*np.where(np.logical_not(map_arr))))
        start_locations = [np.array(loc) for loc in start_locations]
        num_locations = len(start_locations)
        for idx, loc in enumerate(start_locations):
            if verbose and idx % 1 == 0:
                print('sampling trajectories for possible start locations: {}/{}'.format(
                    idx + 1, num_locations))

                trajectories_ = _sample_trajectories(loc, map_arr, cost_map,
                                                     num_trajectories,
                                                     min_dist, max_dist, min_traj_len, diagonal)
                if trajectories_ is not None:
                    trajectories += trajectories_
    return trajectories


def _sample_trajectories(start, map_arr, cost_map, num, min_dist=10, max_dist=20, min_length=None, diagonal=True):
    """ Try to sample num trajectories given a valid start and cost map. """

    # a_star_grid = AStarGrid(cost_map, diagonal=diagonal)

    trajectories = []
    map_arr_tmp = map_arr.copy()
    # get wall locations
    walls = set([tuple(loc) for loc in np.array(np.where(map_arr)).T.tolist()])
    # calculate corners for goal window
    offset = np.ceil(max_dist)
    slices = np.array([(start[i] +
                        np.array([-offset, offset])).astype(int) for i in [0, 1]])
    # make sure first index is not negative
    slices[slices < 0] = 0

    # make sure maximum slice is not out of bounds
    for dim in [0, 1]:
        if slices[dim][1] >= map_arr.shape[dim]:
            slices[dim][1] = map_arr.shape[dim] - 1

    top_left = slices[:, 0]

    slices = [slice(*(slices[i].tolist())) for i in [0, 1]]

    # filter out locations which are not within required distances
    mask = np.zeros(map_arr.shape)
    mask[slices[0], slices[1]] = 1
    locs = np.array(np.where(mask==1)).T.tolist()
    for loc in locs:
        if map_arr[tuple(loc)] == 1:
            continue
        dist = np.linalg.norm(start-np.array(loc))
        if dist < min_dist or dist > max_dist:
            map_arr_tmp[tuple(loc)] = 1

    map_patch = map_arr_tmp[slices[0], slices[1]]

    ends = []
    current_have = 0
    tried = 0

    while True:

        if current_have >= num:
            break

        if tried >= 10 * num:
            break

        random_loc = sample_sg_random(map_patch)

        if random_loc is None:
            # return None to indicate there is no possible end
            # therefore no possible trajectory for this start
            return None

        end = top_left + random_loc
        assert min_dist <= np.linalg.norm(start - end) <= max_dist

        if list(end) in ends:
            tried += 1
            continue

        #t1 = time.time()
        trajectory, values = astar_path(cost_map.astype('float32'), start, end, 1, diagonal)
        if len(trajectory) == 0:
            trajectory = None
        else:
            trajectory = np.insert(trajectory, 0, end, axis=0)[::-1]
        #trajectory, _ = a_star_grid.astar(tuple(start), tuple(end))
        #t2 = time.time()
        #print("calculating path takse {:.3f}".format(t2-t1))

        #t1 = time.time()
        # check whether trajectory goes through walls

        if trajectory is not None:
            traj_traces = [tuple(loc) for loc in trajectory]
            if len(walls.intersection(traj_traces)) > 0:
                trajectory = None
        #t2 = time.time()
        #print(("check going throught walls takse {:.3f}".format(t2-t1)))

        if trajectory is not None:
            if (min_length is None) or (min_length is not None and len(trajectory) > min_length):
                trajectories.append(np.array(trajectory))
                ends.append(list(end))
                current_have += 1
                # print("got one traj")

        tried += 1

    return trajectories

def rescale_trajectory(trajectory, resolution):
    # scale path to real world extent
    return np.array(trajectory + np.array([0.5] * 2)[None, :]) * resolution




