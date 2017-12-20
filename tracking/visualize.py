
""" A wrapper class for visualizing propagation models. """

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils.occ_map_utils import load_map, display_occ_map, plot_grid_map_hmm, show_traj, red_cm, blue_cm, greens_cm
from utils.occ_map_utils import show_map as show_walls
from utils.scene_utils import caculate_traj_from_scene
from data_generator.ped_sim import sample_trajectories
from metrics import cross_entropy
from animation import Plot


map_resolution = 0.2

word_to_vel = { 'UP': [0, 1], 'LEFT': [-1, 0],
                'RIGHT': [1, 0], 'DOWN': [0, -1],
                'UPLEFT': [-1, 1], 'UPRIGHT': [1, 1],
                'DOWNLEFT': [-1, -1], 'DOWNRIGHT': [1, -1]}

# key press to update figure for the next tracking setp
def on_press(event, visualization):
    sys.stdout.flush()
    if event.key == ' ':
        visualization.visualize_next()
        visualization.fig.canvas.draw()

# show an arrow point to the cell clicked
def onclick(event, visualization):

    res = visualization.res
    ix, iy = event.xdata+res/2, event.ydata+res/2

    delta = .3
    for idx in np.ndindex(visualization.axes.shape):
        if visualization.axes[idx] == event.inaxes:
            visualization.axes[idx].arrow(ix+delta, iy+delta, -delta, -delta,
                                          head_width=0.2, head_length=0.1, fc='r', ec='r', zorder=15)

    visualization.fig.canvas.draw()

class Visualize(object):
    """
    A class for visualizing propagation models.
    """
    element_to_title = {"obser": "observation", "pred":"prediction", "corr":"after correction"}

    def __init__(self, models, num_steps, simulated_data=True, tracking=False, measurement_lost=None,
                 dynamically=True, show_map=True, show_seen=False, show_elements=['obser', 'pred', 'corr'],
                 show_colorbar=True, num_col=5, model_names=None, show_at=None, show_metric=True, **kwargs):
        self.models = models
        self.num_models = len(self.models)
        self.map = models[0].map
        self.res = models[0].map_res
        self.extent = models[0].extent
        self.num_steps = num_steps
        self.dynamically = dynamically
        self.simulated_data = simulated_data
        self.tracking = tracking
        self.measurement_lost = measurement_lost
        self.show_map = show_map
        self.show_seen = show_seen
        self.show_metric = show_metric
        show_order = {'pred': 0, 'obser':1, 'corr':2}
        show_elements.sort(key=lambda name: show_order[name])
        self.show_elements = show_elements if tracking else ['pred']
        self.num_col = self._get_num_col(num_col, self.show_elements)
        self.show_at = show_at
        if not self.dynamically and show_at is not None:
            self.num_col = len(show_at)
        self.show_colorbar = show_colorbar
        self.model_names = [model.name for model in self.models]
        if model_names is not None:
            self.model_names = model_names
        self.fig, self.axes, self.plots = self._init_figure()
        # bind key press event to update figure
        self.fig.canvas.mpl_connect('key_press_event', lambda event: on_press(event, self))
        self.fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, self))
        self.initialize_models()
        self.init_plot_contents()

    def initialize_models(self):
        """ This method has to be implemented by subclasses. """
        pass

    def _get_num_col(self, num_col, show_elements):
        """ Determine number of columns in the figure."""
        if self.dynamically:
            all_elements = ['obser', 'pred', 'corr']
            require_show = map(lambda e: e in all_elements, show_elements)
            return sum(require_show)
        else:
            return num_col

    def _init_figure(self):
        num_rows, num_cols = self.num_models, self.num_col
        fig_size = (4 * num_cols, 3.5 * num_rows)
        fig = plt.figure(figsize=fig_size)
        axes = np.empty((num_rows, num_cols), dtype=object)
        plots = np.empty((num_rows, num_cols), dtype=object)
        for i in range(num_rows):
            for j in range(num_cols):
                plot_idx = i * num_cols + j + 1
                ax = fig.add_subplot(num_rows, num_cols, plot_idx)
                axes[i, j] = ax
                # no ticks on axis
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
        return fig, axes, plots
                # plot = Plot(ax, self.map, self.res, plot_seen=self., title='')

    def init_plot_contents(self):
        """ Decide plot contents for every plot."""
        colorbar_dict = {'pred': 'occupancy_axes', 'obser': 'map_axes', 'corr': 'occupancy_axes'}
        for i in range(self.num_models):
            for j in range(self.num_col):

                if not self.show_colorbar:
                    colorbar_on = None
                else:
                    if self.dynamically:
                        colorbar_on = colorbar_dict[self.show_elements[j]]
                    else:
                        colorbar_on = "occupancy_axes" if not self.tracking else None

                self.plots[i, j] = Plot(self.axes[i ,j], self.map, self.res,
                                       colorbar_on=colorbar_on,
                                       plot_map=self.show_map,
                                       plot_seen=self.show_seen,
                                       title='')

                if self.tracking and not self.dynamically:
                    self.add_custom_elements_to_plot(self.plots[i, j])

                if j == 0:
                    self.plots[i, j].set_ylabel(self.model_names[i])

                if self.traj_overlay:
                    # trajectory from scene can only be calculated for one target
                    num_targets = self.num_targets if self.simulated_data else 1
                    self.plots[i, j].add_traj_line(num_targets=num_targets)

        # add title to figure
        if self.simulated_data and self.init_moves is not None:
            self.fig.suptitle("Initial movements = {}".format(str(self.init_moves)))

        # add legend if neccessary
        if self.tracking and not self.dynamically:
            self.add_legend()

    def add_custom_elements_to_plot(self, plot):
        # add false negative axes
        # it shows locations where ground truth is occupied
        # but BOFUM fails to track
        plot.add_custom_image("fn_axes", blue_cm)
        # add true positive axes
        # it shows locations where ground truth is occupied
        # and BOFUM predicts occupancy prob higher than 0
        plot.add_custom_image("tp_axes", greens_cm)
        plot.add_colorbar("tp_axes")
        # add false positive axes
        plot.add_custom_image("fp_axes", red_cm)

    def add_legend(self):
        g_patch = mpatches.Patch(color='g', label='True positive')
        #b_patch = mpatches.Patch(color='b', label='False negative')
        o_patch = mpatches.Patch(color='orange', label='False positive', alpha=.6)
        #handles = [g_patch, b_patch, o_patch]
        handles = [g_patch, o_patch]
        plt.legend(handles=handles, bbox_to_anchor=(.6, 0., .35, .1),
                   loc = 'lower right',
                   ncol=3, mode='expand', borderaxespad=.0,
           bbox_transform=self.fig.transFigure)
        # plt.legend(handles=[g_patch, b_patch, o_patch], bbox_to_anchor=(0., 1.1),
        #            loc=1, ncol=1, mode="expand", borderaxespad=None,
        #             bbox_transform=self.fig.transFigure)

    def start(self):
        # reset model's step counter
        # for model in self.models:
        #     model.t = 0
        plt.show()

    def visualize_next(self):

        if self.dynamically:
            if(self.models[0].t < self.num_steps):

                t = self.models[0].t
                print("t is " + str(t))

                # generate measurement data
                measurement = self.models[0].measurement_at() if self.tracking else None

                for model in self.models:
                    measurement = model.tracking_step(measurement=measurement)

                self.show_dynamically(measurement)
        else:
            self.show_occupancies()

    def show_dynamically(self, observation):

        Ot_pred_max, Ot_pred_min = None, None
        Ot_max, Ot_min = None, None
        if 'pred' in self.show_elements:
            Ot_pred_max = max(map(lambda model: model.P_Ot_pred.max(), self.models))
            Ot_pred_min = min(map(lambda model: model.P_Ot_pred.min(), self.models))

        if 'corr' in self.show_elements:
            Ot_max = max(map(lambda model: model.P_Ot.max(), self.models))
            Ot_min = min(map(lambda model: model.P_Ot.min(), self.models))

        for i, model in enumerate(self.models):
            for j, element in enumerate(self.show_elements):
                if element == 'pred' or element == 'corr':
                    if element == 'pred':
                        Ot, max_, min_ = model.P_Ot_pred, Ot_pred_max, Ot_pred_min
                        ap = model.calc_average_precision(evaluate_prediction=True)
                    else:
                        Ot, max_, min_ = model.P_Ot, Ot_max, Ot_min
                        ap = model.calc_average_precision()

                    title = self.element_to_title[element]

                    if self.tracking and self.show_metric:
                        self.plots[i, j].text.set_text(" Aver. Precision: {:.3f}".format(ap))
                    self.plots[i, j].set_axes_data("occupancy_axes", Ot, min_, max_)
                else:
                    Ot = observation
                    if observation is None:
                        Ot = np.zeros_like(self.map)
                    title = 'observation'
                    self.plots[i, j].set_axes_data("map_axes", Ot)

                if self.show_seen:
                    t = self.models[0].t - 1
                    seen = self.models[0].evaluate_loc_at(t)
                    self.plots[i, j].set_axes_data("seen_axes", seen)

                self.plots[i, j].set_title(title)
                self.plots[i, j].refresh_colorbar()

                if self.simulated_data and self.traj_overlay:
                    self.add_traj_overlay()

    def show_occupancies(self):

        if self.show_at is None:
            gap = self.num_steps // (self.num_col-1)
            self.show_at = [i*gap+1 for i in range(self.num_col)]

        for j, plot_t in enumerate(self.show_at):

            while self.models[0].t < plot_t:
                # generate measurement data
                measurement = self.models[0].measurement_at() if self.tracking else None
                for model in self.models:
                    measurement = model.tracking_step(measurement=measurement)

            occupancies = np.array(map(lambda model: model.P_Ot, self.models))

            Ot_max = occupancies.max()
            Ot_min = occupancies.min()

            for i, model in enumerate(self.models):

                if plot_t > 0 and self.tracking and self.show_metric:
                    x_ent = model.calc_cross_entropy()
                    f1_score = model.calc_f1_score()
                    ap = model.calc_average_precision()
                    self.plots[i, j].text.set_text(" Aver. Precision: {:.3f}".format(ap))
                if not self.tracking:
                    self.plots[i, j].set_axes_data("occupancy_axes", occupancies[i], Ot_min, Ot_max)
                else:
                    self.update_custom_element(i, j, Ot_max)

                self.plots[i, j].refresh_colorbar()

                if i == 0:
                    title = 't={}'.format(plot_t)
                    self.plots[i, j].set_title(title)

                if self.tracking and self.traj_overlay:
                    self.add_traj_overlay(i, j, plot_t)

    def update_custom_element(self, row, col, Ot_max):
        t = self.models[0].t - 1
        model = self.models[row]
        plot = self.plots[row, col]
        occupancy_prob = model.P_Ot
        h_max = occupancy_prob.max()/2
        #occupancy_prob = np.where(occupancy_prob>h_max, occupancy_prob, 0)
        ground_truth = model.ground_truth_at(t)
        overlap = np.logical_and(occupancy_prob, ground_truth)
        # if occupany on ground truth location is higher than 0.1,
        # it is not thought as a false negative
        occupancy_temp = np.where(overlap, occupancy_prob, 0)
        #predicted = np.where(occupancy_temp>0.1, 1, 0)
        false_negative = np.where(occupancy_temp>0.1, 0, ground_truth)
        # if model predicts occupancy higher than 0 on ground truth locations,
        # it is thought as a true positive
        true_positive = np.where(overlap, occupancy_prob, 0)
        # if model predicts occupancy higher than 0 on non-ground truth locations,
        # it is thought as a false positive
        false_positive = occupancy_prob.copy()
        false_positive[overlap] = 0
        # only show for occupancies higher than 1/4 highest occupancy
        # h_max = false_positive.max() / 4
        # false_positive = np.where(false_positive>h_max, false_positive, 0)

        plot.set_axes_data("fn_axes", false_negative, 0, 1)
        plot.set_axes_data("tp_axes", true_positive, 0, Ot_max)
        plot.set_axes_data("fp_axes", false_positive, 0, Ot_max)
        plot.set_axes_data("occupancy_axes", np.zeros_like(occupancy_prob))




class VisualizeRealdata(Visualize):

    def __init__(self, models, num_steps, scene, tracking=True, measurement_lost=None,
                 dynamically=False, show_map=True, show_seen=True, show_elements=['obser', 'pred', 'corr'],
                 show_colorbar=True, num_col=5, model_names=None, show_at=None, show_metric=True, **kwargs):

        self.scene = scene
        self.traj_overlay = kwargs.get('traj_overlay', False)
        if self.traj_overlay:
            self.traj = caculate_traj_from_scene(scene)
        super(VisualizeRealdata, self).__init__(models, num_steps, False, tracking,
                                                  measurement_lost, dynamically, show_map, show_seen,
                                                  show_elements, show_colorbar, num_col, model_names,
                                                  show_at, show_metric, **kwargs)

    def initialize_models(self):
        map(lambda model: model.initialize(self.scene), self.models)

    def add_traj_overlay(self, i, j, t):
        """ Add trajectory lines on plots. """
        plot = self.plots[i, j]
        line = plot.lines[0]
        small_idx = max(0, t-4)
        big_idx = min(t+4, self.num_steps)
        xs, ys = self.traj.T[0][small_idx:big_idx], self.traj.T[1][small_idx:big_idx]
        line.set_data(xs, ys)

class VisualizeSimulation(Visualize):

    def __init__(self, models, num_steps, tracking=True, measurement_lost=None, num_targets=1,
                 dynamically=False, show_map=True, show_elements=['obser', 'pred', 'corr'],
                 show_colorbar=True, num_col=5, model_names=None, diagonal=False, show_at=None,
                 show_metric=True, **kwargs):
        """

        :param models: list
            models to be visualized.
        :param num_steps: int
            number of time steps.
        :param dynamically: bool
            show propagation per time step or show all time steps at once
        :param kwargs:
            show_condition_probs : bool
             if True, show per time step with plots with quiver overlay. This
             option only has effect when dynamically is set to true.
            show_map : bool
                 show map overlay. Default is True.
            num_plot : int
                 number of plots for each propagation model.
            traj_overlay : bool
                 if True, show the sampled trajectory along with propagation.
            init_loc : list
                 a list of coordinate [int, int] of initial locations.
            init_move : list
                 a list of initial movements. Movement could be one of "UP", "DOWN", "LEFT", "RIGHT".
                 Note if init_move or init_loc is not provided, then trajectory
                 is sampled from the map and its initial state is used.
            model_names : list
                 list of model names.
        """

        self.num_targets = num_targets
        self.diagonal = diagonal
        self.current_speed = np.ones((self.num_targets), dtype=int)
        self.steps = np.zeros((self.num_targets), dtype=int)
        self.init_moves = None
        self.kwargs = kwargs
        self.traj_overlay = True
        if 'traj_overlay' in kwargs:
            self.traj_overlay = kwargs['traj_overlay']
        if 'init_locs' in kwargs and 'init_moves' in kwargs:
            tracking = False
            self.traj_overlay = False

        super(VisualizeSimulation, self).__init__(models, num_steps, True, tracking,
                                                  measurement_lost, dynamically, show_map, False,
                                                  show_elements, show_colorbar, num_col, model_names,
                                                  show_at, show_metric, **kwargs)


    def initialize_models(self):
        kwargs = self.kwargs
        if 'init_locs' in kwargs and 'init_moves' in kwargs:
            self.num_targets = len(kwargs['init_locs'])
            Vt0, Ot0 = self.get_initial_state(from_traj=False,
                                                        init_locs=kwargs['init_locs'],
                                                        init_moves=kwargs['init_moves'])

        else:
            self.distances, self.trajs = self.models[0].initialize(self.num_targets, self.num_steps, diagonal=self.diagonal)
            Vt0, Ot0 = self.get_initial_state(from_traj=True)

        for model in self.models:
            if not self.tracking:
                model.initialize(self.num_targets, self.num_steps,
                                 P_Vt=Vt0, P_Ot=Ot0)
            else:
                model.initialize(self.num_targets, self.num_steps,
                                 distances=self.distances, trajectories=self.trajs)


    def _words_to_idx(self, direction):
        """ Transform words of directions to array indices of Vt. """
        extent = self.extent
        idx = np.array([extent // 2, extent // 2])
        for word in direction.split('_'):
            idx += word_to_vel[word]
        idx = np.where(idx >= 0, idx, 0)
        idx = np.where(idx <= extent-1, idx, extent-1)
        return list(idx)

    def _locs_to_idx(self, from_loc, to_loc):
        """ Transform velocity to array indices of Vt. """
        extent = self.extent
        vel_x = to_loc[0] - from_loc[0]
        vel_y = to_loc[1] - from_loc[1]
        vel = np.array([vel_x, vel_y])
        center_idx = np.array([extent // 2, extent // 2])
        return center_idx + vel


    def get_initial_state(self, from_traj, **kwargs):

        Vt0 = np.zeros_like(self.models[0].P_Vt)
        Ot0 = np.zeros_like(self.models[0].P_Ot)

        if not from_traj:
            self.init_moves = kwargs['init_moves']
            init_locs = kwargs['init_locs']
            # indices of initial movements
            mv_idxs = map(self._words_to_idx, kwargs['init_moves'])
        else:
            init_locs = map(lambda traj: traj[0], self.trajs)
            next_locs = map(lambda traj: traj[1], self.trajs)
            mv_idxs = map(self._locs_to_idx, init_locs, next_locs)

        # indices of initial locations
        idxs = np.array(init_locs).T.tolist()
        # occupancy is 1 at init_locs
        Ot0[idxs] = 1.0
        # put locations and movements together
        v_idxs = map(lambda loc, mv_ix: np.concatenate((loc, mv_ix)), init_locs, mv_idxs)
        v_idxs = np.array(v_idxs).T.tolist()
        Vt0[v_idxs] = 1.0
        return Vt0, Ot0

    # def init_plot_contents(self):
    #     """ Decide plot contents for every plot."""
    #     colorbar_dict = {'pred': 'occupancy', 'obser': 'map', 'corr': 'occupancy'}
    #     for i in range(self.num_models):
    #         for j in range(self.num_col):
    #
    #             if not self.tracking:
    #                 colorbar_on = 'occupancy'
    #             else:
    #                 colorbar_on = colorbar_dict[self.show_elements[j]]
    #
    #             self.plots[i, j] = Plot(self.axes[i ,j], self.map, self.res,
    #                                    colorbar_on=colorbar_on,
    #                                    plot_map=self.show_map,
    #                                    title='')
    #             if j == 0:
    #                 self.plots[i, j].set_ylabel(self.model_names[i])
    #
    #             if self.traj_overlay:
    #                 self.plots[i, j].add_traj_line(num_targets=self.num_targets)
    #     # add title to figure
    #     if self.init_moves is not None:
    #         self.fig.suptitle("Initial movements = {}".format(str(self.init_moves)))

    def add_traj_overlay(self, i=None, j=None, t=None):
        """ Add trajectory lines on plots. """
        if t is None:
            t = self.models[0].t - 1
        if t >= 0:
            truncated_trajs = self.models[0].traversed_traj_at()
            if i is not None and j is not None:
                plot = self.plots[i, j]
                for idx, line in enumerate(plot.lines):
                    xs, ys = truncated_trajs[idx].T[0][-5:], truncated_trajs[idx].T[1][-5:]
                    line.set_data(xs, ys)

            else:
                for plot in self.plots.flatten():
                    for idx, line in enumerate(plot.lines):
                        xs, ys = truncated_trajs[idx].T[0][-5:], truncated_trajs[idx].T[1][-5:]
                        line.set_data(xs, ys)


    def show_dynamically_with_condition_probs(self, show_map=True):
        model = None
        for model in self.models:
            if model.__class__.__name__ == 'conditionalPropagateModel':
                model = model
                break
        res = map_resolution
        trans_prob = model.conditional_probs
        fig, axes_ = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(10, 10))
        for i in range(4):
            plot = plt.subplot(2, 2, i + 1)
            probs = np.zeros_like(trans_prob)
            probs[i, ...] = trans_prob[i, ...]
            plot_grid_map_hmm(probs, 'probs', grid_res=res, map_=None, map_res=res)
            directions = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
            if not show_map:
                show_walls(model.P_Ot, resolution=res, cmap='OrRd')
            else:
                show_walls(model.P_Ot, resolution=res, cmap='OrRd', zorder=1)
                show_walls(model.map, resolution=res)
            plot.set_title(directions[i], fontsize=10)

