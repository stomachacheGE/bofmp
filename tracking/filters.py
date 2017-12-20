import os
import imp
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import datetime
import time
import signal
from collections import Counter
from copy import copy

from data_generator.human_mcm import Grid_HMM
from utils.occ_map_utils import load_map, display_occ_map, plot_grid_map_hmm, plot_trajectories, show_map
from utils.plot_utils import plot_4d_tensor
from utils.scene_utils import Scene, animate_scenes
from utils import nd_gaussian
from data_generator.ped_sim import sample_trajectories
from data_generator.map2nnio import rotate
from data_loader import get_map_crop
from test import get_model
from metrics import cross_entropy, tracking_f1_score, tracking_ap_score
from visualize import VisualizeSimulation, VisualizeRealdata
from animation import TrackingAnimSimulation, TrackingAnimRealdata

class TimeoutException(Exception):  # Custom exception class
    pass

def timeout_handler(signum, frame): # Custom signal handler
    print("time out")
    raise TimeoutException

class BOFUM(object):
    """ Class implements Bayesian Occuapancy Filter Using Prior Map Knowledge. """

    def __init__(self, map, omega=0.2, extent=7, name='', noise_var=0.6, map_res=0.2, lost_prob=0,
                 measurement_lost=None, keep_motion=False, window_size=4, motion_keep_factor=0.6,
                 initial_motion_factor=0.5, with_reachability=True, verbose=False):
        self.width, self.height = map.shape
        self.map = map
        self.map_res = map_res
        self.empty_locs = np.where(1-map)
        self.omega = omega
        self.lost_prob = lost_prob
        self.measurement_lost =  measurement_lost
        self.name = name
        self.extent = extent
        self.h_extent = extent // 2
        # mask is used to constrain possible movements
        self.mask = np.ones((extent, extent))
        # self.mask[self.h_extent, self.h_extent] = 0
        # self.mask = np.array([[1, 1, 1],
        #                       [1, 0, 1],
        #                       [1, 1, 1]])
        self.noise_var = noise_var
        self.kernels = None
        self.gaussian_kernel = None
        self.with_reachability = with_reachability
        self.P_reachability = np.zeros((self.width, self.height, extent, extent), dtype=int)
        self.verbose = verbose
        # step counter
        self.t = 0
        self.distances = None
        self.trajectories = None
        # probabilities of velocities in cells
        # We assume that the velocity can only be on four directions,
        # which are constrained by the self.mask:
        #   [0,    UP,  0  ],
        #   [LEFT, 0, RIGHT],
        #   [0,   DOWN, 0  ]
        self.P_Vt = np.zeros((self.width, self.height, extent, extent), dtype=float)
        self.P_Vt_1 = np.zeros_like(self.P_Vt, dtype=float)
        self.P_Vt_pred = np.zeros_like(self.P_Vt, dtype=float)
        # merged velocity probabilities of moving average and predicited velocities
        self.P_Vt_merged = np.zeros_like(self.P_Vt, dtype=float)
        # occupancy probabilities
        self.P_Ot = np.zeros((self.width, self.height), dtype=float)
        self.P_Ot_1 = np.zeros_like(self.P_Ot, dtype=float)
        self.P_Ot_pred = np.zeros_like(self.P_Ot, dtype=float)
        self.P_Z = np.zeros_like(self.P_Ot, dtype=float)
        # once initialized, self.P_T should have dimension of (self.width, self.height, self.width, self.height)
        self.P_T = np.zeros((self.width, self.height, extent, extent))
        self.P_T_reformed = np.zeros_like(self.P_Vt, dtype=float)
        self.P_Ot_reformed = np.zeros_like(self.P_Vt, dtype=float)
        self.Z_last = np.zeros_like(self.map)
        self.keep_motion = keep_motion
        self.window_size = window_size
        self.motion_keep_factor = motion_keep_factor
        self.initial_motion_factor = initial_motion_factor
        self.ma_speeds = np.zeros((self.window_size, extent, extent), dtype=float)
        self.ma_vel = np.zeros((extent, extent), dtype=float)

    def observation_model(self, O, Z):
        """ Apply the observation model.
        O: list. values are either 0(nocc) or 1(occ)
        Z: list. values are either 0(nocc) or 1(occ)
        """
        O = np.array(O)
        Z = np.array(Z)
        equal = O == Z
        correction = np.where(equal, 1-self.omega, self.omega)
        return correction

    def _pos_2_c_idx(self, i, j):
        return i * self.height + j

    def _c_idx_2_pos(self, idx):
        return (idx//self.height, idx%self.height)

    def initialize_reachability(self):
        """
        Initialize reachability matrix.
        """
        # TODO: this initialization is only valid when velocity can only move to neighbor cells
        padded_map = np.pad(self.map, self.h_extent, 'constant', constant_values=0)
        for i in range(self.width):
            for j in range(self.height):
                # this is not accurate since even though two cells are not walls,
                # they can still be unreachable
                self.P_reachability[i, j, :, :] = 1 - (
                    self.map[i, j] or padded_map[i:i+self.extent, j:j+self.extent].copy())

    def refresh_transition(self):
        """
        Refresh transition model.
        """
        self.P_T = self.P_Vt.copy()

        # multiply with mask so that it will only have the constrained movement
        self.P_T *= self.mask

        with np.errstate(divide='ignore', invalid='ignore'):
            self.P_T /= np.sum(self.P_T, axis=(2, 3), keepdims=True)
            self.P_T[~np.isfinite(self.P_T)] = 0

        if self.with_reachability:
            self.P_T *= self.P_reachability

    def prepare_propagations(self):
        """
        For each cell, this function
        calculates its neighbors' occupancy probabilities and
        transition probabilities towards this cell.
        """
        extent, h_extent = self.extent, self.h_extent
        reformed_T = np.zeros((self.width, self.height, extent, extent))
        padded_T = np.pad(self.P_T, ((h_extent, h_extent),(h_extent, h_extent),
                                             (0, 0), (0, 0)), 'constant', constant_values=0)
        for i in range(extent):
            for j in range(extent):
                vel_x, vel_y = i - h_extent, j - h_extent
                reformed_T[:, :, i, j] = padded_T[h_extent-vel_x:h_extent+self.width-vel_x,
                                                  h_extent-vel_y:h_extent+self.height-vel_y,
                                                  i, j]
        self.P_T_reformed = reformed_T

        # get neighbors occupancy
        padded_O = np.pad(self.P_Ot, h_extent, 'constant', constant_values=0)
        for i in range(self.width):
            for j in range(self.height):
                neighbor_occ = padded_O[i:i + extent, j:j + extent]
                # filp up/down and right left, since neighbor who takes
                # right direction is on the left side.
                neighbor_occ = np.fliplr(neighbor_occ)
                neighbor_occ = np.flipud(neighbor_occ)
                self.P_Ot_reformed[i, j, :, :] = neighbor_occ.copy()

    def propagate_velocities(self):
        """
        Propagate velocities based on last velocities and transition model.
        """
        self.P_Vt_1 = self.P_T_reformed * self.P_Ot_reformed

        # velocity probs sum to 1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.P_Vt_1 /= np.sum(self.P_Vt_1, axis=(2, 3), keepdims=True)
            self.P_Vt_1[~np.isfinite(self.P_Vt_1)] = 0

        self.P_Vt_pred = self.P_Vt_1.copy()


        if self.measurement_lost is not None and \
                        self.measurement_lost <= self.t+1 and self.keep_motion:
            self.integrate_past_motion()

        if hasattr(self, 'apply_vel_mask'):
            if self.apply_vel_mask:
                self.integrate_velocity_mask()

        self.P_Vt_merged = self.P_Vt_1.copy()

        if self.noise_var:
            self.P_Vt_1 = self.add_noise()

        if hasattr(self, 'keep_naive_bofum') and self.keep_naive_bofum:
            # gaussians = self.get_gaussian_kernel(num_dim=2)
            # naive_bofum_P_Vt_1 = self.add_noise(self.P_Vt_pred.copy(), gaussians)
            vel = self.P_Vt_pred.copy()
            vel /= np.max(vel, axis=(2, 3), keepdims=True)
            self._integrate_to_velocity(vel, self.naive_bofum_factor, max_mode=True)
            self.P_Vt_merged = self.P_Vt_1.copy()

    def integrate_past_motion(self):

        motion_factor = self.initial_motion_factor * self.motion_keep_factor ** (self.t - self.measurement_lost + 1)
        print("motion factor :{}".format(motion_factor))
        self.ma_vel /= self.ma_vel.max()
        self._integrate_to_velocity(self.ma_vel, motion_factor, max_mode=True)

    def integrate_velocity_mask(self):
        self.vel_mask /= np.max(self.vel_mask, axis=(2, 3), keepdims=True)
        self._integrate_to_velocity(self.vel_mask, self.velocity_mask_factor, max_mode=True)

    def _integrate_to_velocity(self, integrate_vel, factor, max_mode=False):
        occ_cells = np.array(np.where(self.P_Ot)).T
        for (x, y) in occ_cells:
            vel = self.P_Vt_1[x, y]
            if max_mode:
                vel /= vel.max()
            if integrate_vel.ndim == 2:
                self.P_Vt_1[x, y] = factor * integrate_vel + (1 - factor) * vel
            else:
                self.P_Vt_1[x, y] = factor * integrate_vel[x, y] + (1 - factor) * vel

        # velocity probs sum to 1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.P_Vt_1 /= np.sum(self.P_Vt_1, axis=(2, 3), keepdims=True)
            self.P_Vt_1[~np.isfinite(self.P_Vt_1)] = 0

    def add_noise(self, P_Vt=None, condi_probs=None):
        """ Add noise to velocity predictions."""
        if condi_probs is None:
            if self.kernels is None:
                self.construct_kernels()
            condi_probs = self.kernels

        if P_Vt is None:
            P_Vt = self.P_Vt_1

        P_Vt = P_Vt[:, :, :, :, None, None] * condi_probs
        P_Vt = np.sum(P_Vt, axis=(2, 3))

        # velocity probs sum to 1
        with np.errstate(divide='ignore', invalid='ignore'):
            self.P_Vt /= np.sum(self.P_Vt, axis=(2, 3), keepdims=True)
            self.P_Vt[~np.isfinite(self.P_Vt)] = 0

        return P_Vt

    def propagate_occupancy(self):
        """
        Propagate occupancy.
        """
        transition = self.P_T_reformed * self.P_Ot_reformed
        not_occupied = np.prod(1 - transition, axis=(2, 3))
        self.P_Ot_1 = 1 - not_occupied
        self.P_Ot_pred = self.P_Ot_1.copy()

    def _keep_total_occupancy(self):
        """ Utility function used to keep total occupancy the same before and after tracking."""
        total_occupancy = np.sum(self.P_Ot)
        pred_total_occupancy = np.sum(self.P_Ot_1)
        self.P_Ot_1 *= total_occupancy / pred_total_occupancy

    def correction(self, Z):
        """Apply correction step."""
        shape = Z.shape
        Z = Z.flatten()
        O_occ = np.ones_like(Z)
        correction_occ = self.observation_model(O_occ, Z).reshape(shape) * self.P_Ot_1
        correction_nocc = self.observation_model(1-O_occ, Z).reshape(shape) * (1-self.P_Ot_1)
        correction_ = correction_occ + correction_nocc
        self.P_Ot_1  = correction_occ / correction_


    def update_ma_vel(self, Z):

        vel = self.vel_between_measurements(Z)
        # print(vel)
        self.Z_last = Z
        self.ma_speeds[:self.window_size - 1] = self.ma_speeds[1:]
        self.ma_speeds[-1] = vel
        self.ma_vel = np.sum(self.ma_speeds, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.ma_vel /= self.ma_vel.sum()
            self.ma_vel[~np.isfinite(self.ma_vel)] = 0

    def vel_between_measurements(self, Z_next):

        extent = self.extent
        h_extent = extent // 2
        padded_measurement = np.pad(Z_next, ((h_extent, h_extent), (h_extent, h_extent)),
                                    mode='constant', constant_values=0)
        occ_cells = np.array(np.where(self.Z_last)).T
        occ_cells += np.array([h_extent, h_extent])
        v = np.zeros((extent, extent), dtype=float)
        for cell in occ_cells:
            x, y = cell[0], cell[1]
            neighbors = padded_measurement[x - h_extent:x + h_extent + 1,
                        y - h_extent:y + h_extent + 1].copy()
            neighbors[h_extent, h_extent] = 0
            v += neighbors
        with np.errstate(divide='ignore', invalid='ignore'):
            v /= v.sum()
            v[~np.isfinite(v)] = 0
        return v

    def propagate(self, Z=None):
        self.refresh_transition()
        self.prepare_propagations()
        self.propagate_velocities()
        self.propagate_occupancy()
        if Z is not None:
            self.correction(Z.copy())
            self.update_ma_vel(Z.copy())
        self.P_Vt = self.P_Vt_1.copy()
        self.P_Ot = self.P_Ot_1.copy()
        self.t += 1

        if self.verbose:
            print("{} has total occupancy of {}".format(self.name, np.sum(self.P_Ot_1)))
            print("Locations where occupancy probability is higher than 0.5: ")
            locs = np.array(np.where(self.P_Ot_1>0.5)).T.tolist()
            for loc in locs:
                print("P(occupied at {}) = {:.3f}".format(str(loc), self.P_Ot_1[loc[0], loc[1]]))

    def tracking_step(self, **kwargs):

        if "measurement" in kwargs:
            measurement = kwargs['measurement']
        else:
            measurement = self.measurement_at()

        self.propagate(measurement)
        return measurement

    def measurement_at(self):
        measurement = self.ground_truth_at()

        if np.random.rand() < self.lost_prob:
            measurement = None
        elif self.measurement_lost is not None and self.measurement_lost <= self.t:
            measurement = None

        if measurement is None:
            #print("measurement lost")
            pass

        return measurement

    @staticmethod
    def gkern(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """
        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
        return kernel / np.sum(kernel)

    def get_gaussian_kernel(self, num_dim):
        extent = self.extent
        gaussian_kernels = np.zeros(([extent] * num_dim * 2))
        for idx in np.ndindex(*([extent] * num_dim)):
            gaussian_kernels[tuple(idx)] = self.gaussian_kernel_at(idx, num_dim)

        return gaussian_kernels

    def gaussian_kernel_at(self, idx, num_dim=4):
        """ Return a gaussian kernel centered at idx. """
        if self.gaussian_kernel is None:
            increased_extent = self.extent + self.extent // 2 * 2
            self.gaussian_kernel = nd_gaussian([increased_extent] * num_dim,
                                               center=[increased_extent // 2] * num_dim,
                                               cov=self.noise_var)

        slices = map(lambda cor: slice(self.extent-cor-1, self.extent-cor+self.extent-1), idx)
        kernel = self.gaussian_kernel[slices].copy()
        # normalize
        kernel /= np.sum(kernel)
        return kernel

    def calc_cross_entropy(self, evaluate_prediction=True):
        t = self.t-1
        if t < 0:
            raise ValueError("Cannot evaluate cross entropy since tracking not starts yet.")
        ground_truth = self.ground_truth_at(t)
        eval_at = self.evaluate_loc_at(t)
        overlap = np.logical_and(ground_truth, eval_at)
        P_Ot = self.P_Ot_pred if evaluate_prediction else self.P_Ot
        x_ent = cross_entropy(overlap, ground_truth, P_Ot)
        return x_ent

    def calc_f1_score(self, evaluate_prediction=True):
        t = self.t-1
        if t < 0:
            raise ValueError("Cannot evaluate f1 score since tracking not starts yet.")
        ground_truth = self.ground_truth_at(t)
        eval_at = self.evaluate_loc_at(t)
        P_Ot = self.P_Ot_pred if evaluate_prediction else self.P_Ot
        f1_score = tracking_f1_score(eval_at, ground_truth, P_Ot)
        return f1_score

    def calc_average_precision(self, evaluate_prediction=True):
        t = self.t-1
        if t < 0:
            raise ValueError("Cannot evaluate average precision since tracking not starts yet.")
        ground_truth = self.ground_truth_at(t)
        eval_at = self.evaluate_loc_at(t)
        P_Ot = self.P_Ot_pred if evaluate_prediction else self.P_Ot
        score = tracking_ap_score(eval_at, ground_truth, P_Ot)
        return score

class BOFUMSimulation(BOFUM):

    def initialize(self, num_targets, num_steps, P_Vt=None, P_Ot=None,
                   distances=None, trajectories=None, constant_speed=False, diagonal=False):
        """ Initialize the BOFUMSimulation model. """

        self.t = 0

        self.num_targets = num_targets
        self.num_steps = num_steps
        self.constant_speed = constant_speed
        self.diagonal = diagonal
        self.trajectories = trajectories
        self.distances = distances

        # initialize with uniform distribution
        self.P_Vt[...] = 1.0 / (self.extent**2) if P_Vt is None else P_Vt
        self.P_Ot[...] = 1.0 / 2 if P_Ot is None else P_Ot

        self.P_Vt_init = self.P_Vt.copy()
        self.P_Ot_init = self.P_Ot.copy()

        self.initialize_reachability()
        if (P_Vt is None or P_Ot is None) and (distances is None or trajectories is None):
            self.distances, self.trajectories = self.generate_trajectories()
        self.construct_kernels()

        if self.verbose:
            print("Initializing model {}".format(self.name))

        return self.distances, self.trajectories

    def reset(self):
        self.t = 0
        # initialize with uniform distribution
        self.P_Vt = self.P_Vt_init.copy()
        self.P_Ot = self.P_Ot_init.copy()

        if self.verbose:
            print("Reset model.")

    def generate_trajectories(self):
        """ Sample trajectories on the map."""

        # since targets' walking distance defines trajectory length,
        # it is necessary to define targets' distance first
        distances = self._generate_distances()
        # in order to make sure targets are moving at each time step,
        # generate trajectories based on the maximum distance
        traj_length = distances.max()

        # Change the behavior of SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        # sampling traj can take maximum 5s
        signal.alarm(5)

        try:
            trajs = self._sample_traj(traj_length, distances)
        except TimeoutException:
            # regenerate distances and try to sample trajectories again
            distances, trajs = self.generate_trajectories()

        # reset the alarm
        signal.alarm(0)

        return distances, trajs

    def _generate_distances(self):
        """ Generate a distance matrix which stores how far targets are away
         from starting point at each time step. """
        speeds_shape = (self.num_targets, self.num_steps)
        if self.constant_speed:
            speeds = np.ones(speeds_shape, dtype=int)
        else:
            speeds = np.zeros(speeds_shape, dtype=int)
            # initial speed for all targets is 1
            current_speed = np.ones(self.num_targets, dtype=int)
            # for each time step, target may have acceleration
            # normally dirstributed around mean 0 and var of self.noise_var
            std = np.sqrt(self.noise_var)
            acc = np.around(np.random.normal(0, std, speeds_shape)).astype(int)
            for i in range(self.num_steps):
                speeds[..., i] = current_speed
                current_speed += acc[..., i]
            # ensure speed is within range 1 <= speed <= max_vel
            max_vel = self.extent // 2
            speeds = np.clip(speeds, 1, max_vel)
        distances = np.zeros_like(speeds)
        # caculate distance until each time step based on speeds
        for t in range(self.num_steps):
            distances[:, t] = np.sum(speeds[:, :(t+1)], axis=1)
        return distances

    def _sample_traj(self, traj_length, distances):
        trajectories = sample_trajectories(self.map, self.num_targets,
                                   diagonal=self.diagonal, mode='random',
                                   min_dist=int(traj_length/np.sqrt(2)), max_dist=1000,
                                   min_traj_len=traj_length,
                                   verbose=True)
        # clip trajectories to the same length and incorporate speeds
        trajectories_ = np.array(map(lambda traj, distance: traj[distance], trajectories, distances))
        # detect whether objects collide, if so, raise exception
        # so that it can resample trajectories
        for t in range(self.num_steps):
            locations = trajectories_[:, t]
            if self._collision_detected(locations):
                raise TimeoutException

        return trajectories

    def _collision_detected(self, locations):
        # if any of the locations has distance within 3 cells,
        # they are thought to collide
        for idx_1, loc_1 in enumerate(locations):
            for idx_2, loc_2 in enumerate(locations):
                if idx_1 != idx_2:
                    if np.linalg.norm(loc_1-loc_2) <= 2:
                        print("collision detected")
                        return True
        return False


    def _uniform_occupancy(self):
        """
        Caculate occupancy probability assuming occupancy is uniformally
        distributed over all empty cells in the map.
        """
        empty_cells = np.logical_not(self.map)
        num_free_locs = np.sum(empty_cells)
        free_locs = np.where(empty_cells)
        P_O = np.zeros((self.width, self.height), dtype=float)
        P_O[free_locs] = float(self.num_targets) / num_free_locs
        return P_O

    def ground_truth_at(self, t=None):
        """The ground truth occupancy at time step t."""
        if self.distances is None or self.trajectories is None:
            raise ValueError("Ground truth is not available since"
                             " no trajecotry is found.")
        if t is None:
            t = self.t
        ground_truth = np.zeros_like(self.map)
        # add target current position
        for i in range(self.num_targets):
            distance_so_far = self.distances[i, t]
            cur_pos = tuple(self.trajectories[i][distance_so_far-1])
            ground_truth[cur_pos] = 1
        return ground_truth

    def targets_loc_at(self, t=None):
        """Targets' locations at time step t."""
        if self.distances is None or self.trajectories is None:
            raise ValueError("Targets' location is not available since"
                             " no trajecotry is found.")
        if t is None:
            t = self.t
        targets_loc = []
        # add target current position
        for i in range(self.num_targets):
            distance_so_far = self.distances[i, t]
            cur_pos = tuple(self.trajectories[i][distance_so_far - 1])
            targets_loc.append(cur_pos)
        return targets_loc

    def evaluate_loc_at(self, t=None):
        """ Locations where cross entropy should be evaluated. """
        # evaluate on non-wall locations no matter which time step
        return 1-self.map

    def traversed_traj_at(self, t=None):
        """ Trajectories that has been tranversed until time t."""
        t = self.t if t is None else t
        truncate = lambda traj, distance: (traj[:distance]+np.array([0.5, 0.5])) * self.map_res
        truncated_trajs = np.array(map(truncate, self.trajectories, self.distances[..., t-1]))
        return truncated_trajs

    def show_trajectories(self, random=False):
        trajs = self.trajectories
        if random:
            trajs = [self.trajectories[np.random.randint(0, self.num_targets)]]
        plot_trajectories(trajs, self.map, self.map_res)

    def trajs_to_scene(self):

        walls = set(map(lambda point: tuple(point), np.array(np.where(self.map)).T.tolist()))
        hits = np.zeros((self.num_steps,) + self.map.shape)
        # make a single point on the trajectory a square with width of 3 cells
        # the first element of occupied cells for each time step is the center
        distances = [(x, y) for x in [0, -1] for y in [0, -1]]
        w, h = self.map.shape

        # clip trajectories to the same length and incorporate speeds
        trajectories_ = np.array(map(lambda traj, distance: traj[distance], self.trajectories, self.distances))

        for idx_1, traj in enumerate(self.trajectories_):
            for idx_2, point in enumerate(traj):
                def get_neighbor(dist):
                    x = min(max(0, point[0] + dist[0]), w - 1)
                    y = min(max(0, point[1] + dist[1]), h - 1)
                    return (x, y)

                neighbors = set(map(get_neighbor, distances))
                neighbors = neighbors.difference(neighbors.intersection(walls))
                for neighbor in neighbors:
                    hits[(idx_2,) + neighbor] = 1
        t = 0.25 * self.num_steps
        seens = np.repeat(np.logical_not(self.map)[None, :, :], self.num_steps, axis=0)
        scene = Scene(self.map, self.map_res, [0, 0], 0, t, hits, seens)
        return scene



class BOFUMRealdata(BOFUM):

    def initialize(self, scene, preprocessing=True):
        """ Initialize the BOFUMReal model. """

        self.t = 0
        self.scene = scene
        if preprocessing:
            self.scene = self.scene_preprocessing(self.scene)
        # initialize with uniform distribution
        self.P_Vt[...] = 1.0 / (self.extent**2)
        self.P_Ot[...] = 1.0 / 2

        self.initialize_reachability()
        self.construct_kernels()

        if self.verbose:
            print("Initializing model {}".format(self.name))

    def reset(self):

        self.t = 0
        # initialize with uniform distribution
        self.P_Vt[...] = 1.0 / (self.extent**2)
        self.P_Ot[...] = 1.0 / 2

        if self.verbose:
            print("Reset model.")

    def update(self, scene):

        print("update model %s" % self.name)
        self.map = scene.static_map
        if isinstance(self, conditionalBOFUM):
            self.nn_probs = self.get_nn_probs(None)
        self.initialize(scene)

    def ground_truth_at(self, t=None):
        """The ground truth occupancy at time step t."""
        if t is None:
            t = self.t
        ground_truth = self.scene.hits[t].copy()
        return ground_truth

    def evaluate_loc_at(self, t=None):
        """ Specify locations where cross entropy should be caculated. """
        if t is None:
            t = self.t
        # evaluate on scene's seen area, but excluding walls, for each time step
        seen = self.scene.seens[t].copy()
        # exclude walls
        intersection = np.where(np.logical_and(seen, self.map))
        seen[intersection] = 0
        return seen

    @staticmethod
    def scene_preprocessing(scene):
        t = scene.hits.shape[0]
        # get all occupied cells
        counts = Counter()
        for hit in scene.hits:
            occ_cells = np.array(np.where(hit)).T.tolist()
            for cell in occ_cells:
                counts[tuple(cell)] += 1
        static_cells = []
        for cell in counts:
            # if a cell appears over 1/3 of the time,
            # it is thought as a static cell
            if counts[cell] > t/4:
                static_cells.append(cell)
        idxs = np.array(static_cells).T.tolist()

        map_ = scene.static_map
        width, height = map_.shape
        padded_map = np.pad(map_, ((1, 1), (1, 1)), 'constant', constant_values=0)

        if len(idxs) > 0:
            for hit in scene.hits:
                hit[idxs] = 0
                for i in np.arange(-1, 2):
                    for j in np.arange(-1, 2):
                        # exclude observations near walls
                        translated_map = padded_map[1+i:1+i+width, 1+j:1+j+height]
                        intersection = np.where(np.logical_and(hit, translated_map))
                        hit[intersection] = 0

        # if a frame dose not have any occupancy,
        # make it the same as last frame
        for i in range(scene.hits.shape[0]):
            if not np.any(scene.hits[i]):
                #print("frame caught at %d" % i)
                if i == 0:
                    next_ = 1
                    while next_ < len(scene.hits) - 1 and not np.any(scene.hits[next_]):
                        next_ += 1
                    scene.hits[0] = scene.hits[next_]
                    #print("replace with frame %d" % next_)
                    continue
                scene.hits[i] = scene.hits[i-1]

        # remove noisy observations
        num_neighbors = 3
        for i in range(scene.hits.shape[0]):
            hit = scene.hits[i]
            locs = np.array(np.where(hit)).T
            for (x, y) in locs:
                x_s, y_s = x - num_neighbors, y - num_neighbors
                x_l, y_l = x + num_neighbors + 1, y + num_neighbors + 1
                x_s, x_l = max(0, min(x_s, width)), max(0, min(x_l, width))
                y_s, y_l = max(0, min(y_s, width)), max(0, min(y_l, width))
                if np.sum(hit[x_s:x_l, y_s:y_l]) == 1 and np.sum(hit) >1:
                    #print("hit[{}:{},{}:{}].sum()=={}".format(x_s, x_l, y_s, y_l, np.sum(hit[x_s:x_l, y_s:y_l])))
                    #print("observation at loc (%d, %d) frame (%d) is noise" % (x, y, i))
                    hit[x, y] = 0

        return scene

class naiveBOFUM(BOFUMSimulation, BOFUMRealdata):
    """
    A simple propagation model that implements linear transition.
    """
    def __init__(self, map_, omega=0.2, extent=7, name='', noise_var=None,
                 map_res=0.2, lost_prob=0, measurement_lost=None, keep_motion=False,
                 window_size=4, keep_motion_factor=0.6, initial_motion_factor=0.5,
                 simulated_data=True, with_reachability=True, verbose=False):
        self.simulated_data = simulated_data
        super(naiveBOFUM, self).__init__(map_, omega, extent, name, noise_var, map_res, lost_prob,
                                         measurement_lost, keep_motion, window_size,
                                         keep_motion_factor, initial_motion_factor, with_reachability, verbose)

    def construct_kernels(self):
        """ Construct gaussian kernels for adding noise."""
        self.kernels = self.get_gaussian_kernel(num_dim=2)

        # # other movements cannot stay
        # self.kernels[..., self.h_extent, self.h_extent] = 0
        # # staying occupancy always stay
        # staying_v = np.zeros((self.extent, self.extent), dtype=float)
        # staying_v[self.h_extent, self.h_extent] = 1
        # self.kernels[self.h_extent, self.h_extent] = staying_v

        # normalize to a probability distribution
        with np.errstate(divide='ignore', invalid='ignore'):
            self.kernels /= self.kernels.sum(axis=(2, 3), keepdims=True)
            self.kernels[~np.isfinite(self.kernels)] = 0

        if self.verbose:
            plot_4d_tensor(self.kernels, title=self.name)

    def initialize(self, *args, **kwargs):
        """ Initialize differently according to different data source."""
        if self.simulated_data:
            return BOFUMSimulation.initialize(self, *args, **kwargs)
        else:
            BOFUMRealdata.initialize(self, *args, **kwargs)

    def reset(self, *args, **kwargs):
        if self.simulated_data:
            BOFUMSimulation.reset(self, *args, **kwargs)
        else:
            BOFUMRealdata.reset(self, *args, **kwargs)

    def update(self, *args, **kwargs):
        if self.simulated_data:
            raise AttributeError("Method BOFUMSimulation.update is not implemented.")
        else:
            BOFUMRealdata.update(self, *args, **kwargs)

    def ground_truth_at(self, t=None):
        """ Get ground truth differently according to different data source."""
        if self.simulated_data:
            return BOFUMSimulation.ground_truth_at(self, t)
        else:
            return BOFUMRealdata.ground_truth_at(self, t)

    def evaluate_loc_at(self, t=None):
        """ Get locations where evaluation should be done differently according to different data source."""
        if self.simulated_data:
            return BOFUMSimulation.evaluate_loc_at(self, t)
        else:
            return BOFUMRealdata.evaluate_loc_at(self, t)

class conditionalBOFUM(naiveBOFUM):
    """
    A propagation model that uses conditional probability predictions from the CNN network.
    """

    def __init__(self, map_, model_name='', omega=0.2, extent=7, name='', nn_probs=None, noise_var=None,
                 map_res=0.2, lost_prob=0, measurement_lost=None, keep_motion=False,
                 window_size=4, keep_motion_factor=0.8, initial_motion_factor=0.5,
                 simulated_data=True, with_reachability=True,
                 force_predict=False, fake_network_pred=False,
                 velocity_mask=False, velocity_mask_factor=0.5,
                 keep_naive_bofum=False, naive_bofum_factor=0.5,
                 acceleration_interpretation=True, blur_spatially=False, blur_extent=9,
                 blur_var=1.2, verbose=False):
        super(conditionalBOFUM, self).__init__(map_, omega, extent, name, noise_var,
                                               map_res, lost_prob, measurement_lost, keep_motion,
                                               window_size, keep_motion_factor, initial_motion_factor,
                                               simulated_data, with_reachability, verbose)
        self.model_name = model_name
        self.force_predict = force_predict
        self.acceleration_interpretation = acceleration_interpretation

        self.config = self.get_cf()
        self.conditional = self.config.conditional_prob
        self.apply_vel_mask = velocity_mask
        self.vel_mask = None
        self.velocity_mask_factor = velocity_mask_factor
        self.keep_naive_bofum = keep_naive_bofum
        self.naive_bofum_factor = naive_bofum_factor
        self.num_directions = self.config.num_directions
        self.fake_nn_pred = fake_network_pred
        self.blur_spatially = blur_spatially
        self.blur_extent = blur_extent
        self.blur_var = blur_var
        self.cnn_model = None
        self.nn_probs = self.get_nn_probs(nn_probs)

    def get_cf(self):
        parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cf_path = os.path.join(parent_folder, 'trained_models', self.model_name, 'cf.py')
        # print(cf_path)
        config = imp.load_source('config_{}'.format(id(self)), cf_path)
        return config


    def get_nn_probs(self, nn_probs):

        if self.fake_nn_pred:
            nn_probs = self.get_fake_probs(self.conditional, self.config)
            return nn_probs

        # get output from neural network
        if nn_probs is None:
            nn_probs = self.get_cnn_output()

        # reshape nn_probs to shape (width, height, 3, 3, 3, 3)
        nn_probs = self.reshape_probs(nn_probs, self.conditional, self.config)

        if self.blur_spatially:
            if self.conditional:
                print("blurring...")
                nn_probs = self.blur_probs_spatially(nn_probs, self.blur_extent, self.blur_var)
            else:
                print("Blurring spatially is not implemented for joint probabilites.")

        return nn_probs


    def get_fake_probs(self, conditional, config):
        nn_probs = np.zeros((self.width, self.height, 3, 3, 3, 3))
        transition_per_cell = np.zeros((3, 3, 3, 3))
        idx_1 = np.tile(np.arange(3), 3)
        idx_2 = np.repeat(np.arange(3), 3)
        constant = 1 if conditional else 1. / config.num_directions
        transition_per_cell[idx_1, idx_2, idx_1, idx_2] = constant
        nn_probs[:, :, ...] = transition_per_cell
        return nn_probs

    def get_cnn_output(self):
        """
        Get the conditional probabilities from the trained model.
        """
        trained_models_path = self.config.trained_model_path
        model_path = trained_models_path + self.model_name
        save_path = '{}/{}_{}_output.npy'.format(model_path, self.width, self.height)
        if os.path.isfile(save_path) and (not self.force_predict):
            probs = np.load(save_path)
        else:
            if self.cnn_model is None:
                self.cnn_model = get_model(model_path)
            # probs has shape (num_channels, width, height)
            probs = self.cnn_model(self.map[None, None, :, :])[0]
            np.save(save_path, probs)
        return probs

    def reshape_probs(self, nn_probs, condi_prob, config):
        """
        Reshape probs to (width, height, 3, 3, 3, 3).
        """
        desired_shape = (self.width, self.height, 3, 3, 3, 3)
        if nn_probs.shape == desired_shape:
            return nn_probs

        if condi_prob:
            # if shape is (n_channels, width, height)
            if nn_probs.ndim == 3:
                nn_probs = np.transpose(nn_probs, axes=(1, 2, 0))
                shape = (self.width, self.height, self.num_directions, self.num_directions)
                nn_probs = nn_probs.reshape(*shape)

            # otherwise, shape is (width, height, num_directions, num_directions)
            velocities = config.velocities

            v_idxs = [Grid_HMM.two_d_vel_to_idx(vel) for vel in velocities]
            probs = np.zeros((self.width, self.height, 3, 3, 3, 3))
            for i in range(self.num_directions):
                for j in range(self.num_directions):
                    probs[:, :, v_idxs[i][0], v_idxs[i][1],
                                v_idxs[j][0], v_idxs[j][1]] = nn_probs[:, :, i, j]
        else:
            vel_idxs = config.unique_vel_idxs
            vel_idxs_b = config.unique_vel_idxs_backward
            num_vel = len(vel_idxs)
            if nn_probs.shape == (num_vel, self.width, self.height):
                nn_probs = np.transpose(nn_probs, axes=(1, 2, 0))
            probs = np.zeros((self.width, self.height, 3, 3, 3, 3))
            for i in range(num_vel):
                # multiply with 0.5 since each entry is the sum
                # of prob of that vel and its backward
                vel_idx, vel_idx_b = vel_idxs[i], vel_idxs_b[i]
                probs[:, :, vel_idx[0], vel_idx[1],
                            vel_idx[2], vel_idx[3]] = 0.5 * nn_probs[:, :, i]
                probs[:, :, vel_idx_b[0], vel_idx_b[1],
                            vel_idx_b[2], vel_idx_b[3]] = 0.5 * nn_probs[:, :, i]
        return probs

    def blur_probs_spatially(self, nn_probs, blur_extent=9, var=1.2):

        idxs = map(lambda vel: Grid_HMM.two_d_vel_to_idx(vel), self.config.velocities)
        turns = []
        for idx_xy in np.ndindex(self.map.shape):
            for idx_vel_last in idxs:
                max_idx = np.argmax(nn_probs[idx_xy + idx_vel_last])
                max_idx = np.unravel_index(max_idx, (3, 3))
                if not max_idx == idx_vel_last:
                    turns.append(idx_xy + idx_vel_last)
        print("found")

        w, h = nn_probs.shape[:2]
        half_be = blur_extent // 2

        blurred = nn_probs.copy()
        blurred = np.pad(blurred, ((half_be, half_be), (half_be, half_be),
                                   (0, 0), (0, 0), (0, 0), (0, 0)),
                                    mode='constant', constant_values=0)

        gaussian = nd_gaussian((blur_extent, blur_extent), (half_be, half_be), var)

        for turn in turns:
            temp = gaussian[..., None, None] * nn_probs[turn]
            x, y = turn[0], turn[1]
            vel_x, vel_y = turn[2], turn[3]
            blurred[x:x+blur_extent, y:y+blur_extent, vel_x, vel_y] += temp
        blurred = blurred[half_be:half_be + w, half_be:half_be + h]

        # normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            blurred /= blurred.sum(axis=(4, 5), keepdims=True)
            blurred[~np.isfinite(blurred)] = 0

        return blurred

    def construct_kernels(self):
        """Construct kernels. For naiveBOFUM, kernels act as applying noise.
        For conditionalBOFUM, kernels act as motion model of humans. """

        # pad condi_probs to extent
        probs = self.nn_probs
        pad_ = (self.extent - probs.shape[-1]) / 2
        probs = np.pad(probs, ((0, 0), (0, 0),
                                           (pad_, pad_), (pad_, pad_),
                                           (pad_, pad_), (pad_, pad_)),
                                            'constant', constant_values=0)

        self.kernels = np.zeros_like(probs)
        if self.noise_var is None:
            self.noise_var = 0.00001

        t1 = time.time()
        gaussians = self.get_gaussian_kernel(num_dim=4)
        t2 = time.time()
        if self.verbose:
            print("generating gaussian kernels takes {} seconds".format(t2 - t1))

        t1 = time.time()
        for i in range(self.width):
            for j in range(self.height):
                temp = probs[i, j]
                if self.acceleration_interpretation:
                    verbose = False
                    if i == 50 and j == 50:
                        verbose = True
                    self.kernels[i, j] = conditionalBOFUM.blur_on_acceleration(temp, self.extent, gaussians, verbose)
                else:
                    temp_blurred = np.zeros_like(temp)
                    non_zero_idxs = np.array(np.where(temp)).T.tolist()
                    for idx in non_zero_idxs:
                        temp_blurred += temp[tuple(idx)] * gaussians[tuple(idx)]
                    self.kernels[i, j] = temp_blurred

        if self.apply_vel_mask:
            joint_P_V = self.kernels.copy()
            self.vel_mask = np.sum(joint_P_V, axis=(4, 5))
            # velocity probs sum to 1
            with np.errstate(divide='ignore', invalid='ignore'):
                self.vel_mask /= np.sum(self.vel_mask, axis=(2, 3), keepdims=True)
                self.vel_mask[~np.isfinite(self.vel_mask)] = 0

        # normalize to a conditional probability distribution, i.e., conditioned on last movement
        with np.errstate(divide='ignore', invalid='ignore'):
            self.kernels /= self.kernels.sum(axis=(4, 5), keepdims=True)
            self.kernels[~np.isfinite(self.kernels)] = 0

        t2 = time.time()
        if self.verbose:
            print("caculating kernels takes {} seconds.".format(t2 - t1))

        if self.verbose:
            idx = np.random.randint(0, self.empty_locs[0].shape[0])
            x, y = self.empty_locs[0][idx], self.empty_locs[1][idx]
            plot_4d_tensor(self.kernels[x, y], title="{} at ({}, {})".format(self.name, x, y))

    @staticmethod
    def blur_on_acceleration(v, extent, gaussians, verbose=False):
        """ Calculate acceleration from network output and apply Gaussian
        blur in acceleration space. Then convert back to velocity space."""
        n_nn_vel = v.shape[0]
        v_nn_max = (n_nn_vel - 1) // 2

        # calculate accelerations
        # padding "to" dimension with maximum velocity, i.e., 1
        v_padded = np.pad(v, ((0, 0), (0, 0),
                              (v_nn_max, v_nn_max), (v_nn_max, v_nn_max)),
                               mode='constant', constant_values=0)
        a = np.zeros_like(v)
        for x, v_x in enumerate(np.arange(-v_nn_max, v_nn_max + 1)):
            for y, v_y in enumerate(np.arange(-v_nn_max, v_nn_max + 1)):
                a[x, y] = v_padded[x, y,
                          v_nn_max + v_x:v_nn_max + n_nn_vel + v_x,
                          v_nn_max + v_y:v_nn_max + n_nn_vel + v_y]

        if verbose:
            plot_4d_tensor(a, 'before manipulation')

        #modify acceleration
        # idxs_vels = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        # idxs = idxs_vels + np.array([extent//2, extent//2])
        # for i, idx_1 in enumerate(idxs):
        #     #print(idx_1)
        #     for vel_2 in np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]):
        #         idx_2 = np.array([extent//2, extent//2]) + vel_2
        #         change = [0, 0]
        #         change[list(vel_2).index(0)] = idxs_vels[i][list(vel_2).index(0)]
        #         idx_2_changed = idx_2 + change
        #         #print("{} -> {}".format(idx_2, idx_2_changed))
        #         temp = a[tuple(idx_1) + tuple(idx_2)]
        #         a[tuple(idx_1)+tuple(idx_2_changed)] = a[tuple(idx_1)+tuple(idx_2)]
        #         a[tuple(idx_1) + tuple(idx_2)] = 0

        if verbose:
            plot_4d_tensor(a, 'after manipulation')

        n_vel = extent
        v_max = n_vel // 2
        n_acc = n_vel
        acc_max = n_acc // 2
        v_max_accelarated = v_max + acc_max

        # blur acceleration
        pad = n_vel // 2 - n_nn_vel // 2
        a = np.pad(a, ((pad, pad), (pad, pad), (pad, pad), (pad, pad)),
                        mode='constant', constant_values=0)
        non_zero_idxs = np.array(np.where(a)).T.tolist()
        a_blurred = np.zeros_like(a)

        for idx in non_zero_idxs:
            idx_ = tuple(idx)
            a_blurred += a[idx_] * gaussians[idx_]

        a = a_blurred

        # restore to velocity space
        v_new = np.zeros_like(a)
        pad = v_max_accelarated - v_max
        a_padded = np.pad(a, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                             mode='constant', constant_values=0)
        for x, v_x in enumerate(np.arange(-v_max, v_max + 1)):
            for y, v_y in enumerate(np.arange(-v_max, v_max + 1)):
                v_new[x, y] = a_padded[x, y,
                              pad - v_x:pad + n_vel - v_x,
                              pad - v_y:pad + n_vel - v_y]

        return v_new


