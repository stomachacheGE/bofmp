
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import ticker
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
import numpy as np

from utils.occ_map_utils import load_map, display_occ_map, plot_grid_map_hmm, show_traj, \
    black_cm, green_cm, red_cm, blue_cm, greens_cm, greys_cm
from utils.occ_map_utils import show_map
from utils.plot_utils import plot_4d_tensor

# key press to update figure for the next tracking setp
def on_press(event, animation):
    if event.key == ' ':
        if animation.pause:
            animation.event_source.stop()
        else:
            animation.event_source.start()
        animation.pause ^= True

def onclick(event, anim):
    models = anim.models
    ix, iy = event.xdata, event.ydata
    coords = np.floor(np.array([ix, iy]) / models[0].map_res).astype(int)
    print("Click at coordinates: {}".format(coords))

    all_axes = [plot.axes for plot in anim.plots]

    for i, ax in enumerate(all_axes):
        # For infomation, print which axes the click was in
        if ax == event.inaxes:
            #print "Click is at filter {}".format(anim.models[i].name)
            break

    clicked = np.zeros_like(models[0].map)
    x, y = coords[0], coords[1]
    clicked[x, y] = 1
    for plot in anim.plots:
        plot.set_axes_data("occupancy_axes", clicked)
    anim.fig.canvas.draw()

    accessories_figures = [anim.nn_output_fig, anim.kernel_fig]
    for fig in accessories_figures:
        if fig is not None:
            fig.clear()
    anim.nn_output_fig.suptitle('Network Output')
    anim.kernel_fig.suptitle('Motion pattern')

    if models[i].kernels.ndim == 6:
        kernel = models[i].kernels[x, y]
        condi_prob = models[i].nn_probs[x, y]
    else:
        kernel = models[i].kernels
        condi_prob = None

    plot_4d_tensor(kernel, fig=anim.kernel_fig)
    if condi_prob is not None:
        plot_4d_tensor(condi_prob, fig=anim.nn_output_fig)

    anim.set_axis_ticks(models[i].extent)
    anim.accessories_plots['ma_plot'].set_axes_data("occupancy_axes", np.ones_like(models[i].ma_vel))
    anim.accessories_plots['vel_plot'].set_axes_data("occupancy_axes", models[i].P_Vt_pred[x, y])
    anim.accessories_plots['merge_vel_plot'].set_axes_data("occupancy_axes", models[i].P_Vt_merged[x, y])
    anim.accessories_plots['final_vel_plot'].set_axes_data("occupancy_axes", models[i].P_Vt[x, y])

    accessories_figures += [anim.vel_fig]
    for fig in accessories_figures:
        if fig is not None:
            fig.canvas.draw()

    for k, plot in anim.accessories_plots.items():
        plot.refresh_colorbar()

    model_names = map(lambda model: model.name, models)
    occs = map(lambda model: model.P_Ot[x, y], models)
    for name, occ in zip(model_names, occs):
        print("loc ({}, {}) of model {} has occupancy of {}".format(x, y, name, occ))

class Plot(object):

    def __init__(self, axes, map, res, plot_map=True, plot_seen=False, show_text=True, colorbar_on=None, title=None):
        self.axes = axes
        self.map = map
        self.res = res
        self.plot_seen = plot_seen
        self.plot_map = plot_map
        self.map_axes = None
        self.occupancy_axes = None
        self.ground_truth_axes = None
        self.seen_axes = None
        self.colorbars = []
        self.show_text = show_text

        if title is None:
            title = 'Measurements'
        self.axes.set_title(title)

        if show_text:
            self.text = self.axes.text(0.92, 0.92, "", bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5},
                        transform=self.axes.transAxes, ha="right", color='white', zorder=14)

        self.add_images()
        self.add_colorbar(colorbar_on)

    def add_images(self):
        """Add AxesImages for showing map, occupancy and seen."""
        occupancy = np.zeros(self.map.shape, dtype=float)
        self.occupancy_axes = show_map(occupancy, self.res, cmap=red_cm, ax=self.axes, zorder=11)
        # initialize plots with map
        map_ = self.map if self.plot_map else np.zeros_like(self.map)
        self.map_axes = show_map(map_, self.res, cmap=black_cm, ax=self.axes, zorder=12)
        if self.plot_seen:
            # add seen image
            self.seen_axes = show_map(occupancy, self.res, cmap=black_cm, alpha=0.2, ax=self.axes)

    def set_axes_data(self, axes_name, data, vmin=None, vmax=None):
        image_ax = getattr(self, axes_name)
        image_ax.set_data(np.rot90(data))
        vmin = vmin if vmin is not None else data.min()
        vmax = vmax if vmax is not None else data.max()
        image_ax.set_clim([vmin, vmax])

    def add_custom_image(self, axes_name, cmap=None, image=None, **kwargs):
        if image is None:
            image = np.zeros(self.map.shape, dtype=float)

        image_ax = show_map(image, self.res, cmap=cmap, ax=self.axes, **kwargs)
        setattr(self, axes_name, image_ax)

    def add_colorbar(self, colorbar_on):
        if colorbar_on is None:
            return

        image_axes = getattr(self, colorbar_on)

        if image_axes is not None:
            cb = plt.colorbar(image_axes, ax=self.axes, fraction=0.046, pad=0.04)
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb.locator = tick_locator
            self.colorbars.append(cb)

    def set_ylabel(self, text='', **kwargs):
        self.axes.set_ylabel(text, **kwargs)

    def refresh_colorbar(self):
        for cb in self.colorbars:
            cb.update_ticks()

    def add_traj_line(self, num_targets=1):
        """ Add 2D Lines for showing trajectories."""
        colors = cm.Dark2(np.linspace(0, 1, num_targets))
        # add lines for showing trajectories
        self.lines = map(lambda _: self.axes.add_line(Line2D([], [], zorder=14, color='grey')), range(num_targets))

    def set_title(self, title):
        self.axes.set_title(title)

    def set_text(self, text):
        self.text.set_text(text)

class TrackingAnimation(animation.TimedAnimation):

    def __init__(self, models, num_steps, simulated_data, plot_seen=False, plot_map=True, show_text=True, accessories=None):
        self.num_models = len(models)
        self.models = models
        self.map = models[0].map
        self.res = models[0].map_res
        self.num_steps = num_steps
        self.simulated_data = simulated_data
        self.show_map = plot_map
        self.show_seen = plot_seen
        self.show_text = show_text
        self.nn_output_fig = None
        self.kernel_fig = None
        self.vel_fig = None
        self.accessories_plots = None
        self.accessories = accessories
        self.initialize_figure()
        self.initialize_models()
        self.initialize_accessories()
        print(self.accessories_plots)
        self.fig.canvas.mpl_connect('key_press_event', lambda event: on_press(event, self))
        self.fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, self))

        animation.TimedAnimation.__init__(self, self.fig, interval=500, blit=True, repeat=True, repeat_delay=1000)

    def initialize_figure(self):
        fig_size = (5 * self.num_models, 5)
        self.fig = plt.figure(figsize=fig_size)
        self.pause = True
        # bind key press event to pause animation
        self.plots = []
        for i in range(self.num_models):
            axes = self.fig.add_subplot(1, self.num_models, i + 1)
            title = self.models[i].name
            colorbar_on = "occupancy_axes" if self.simulated_data else None
            plot = Plot(axes, self.map, self.res, self.show_map, self.show_seen, self.show_text, colorbar_on, title=title)
            self.add_custom_element(plot)
            self.plots.append(plot)
        self.fig_title_axes = self.fig.add_axes([.4, .9, .2, .05])
        self.fig_title_axes.set_axis_off()
        self.fig_title = self.fig.text(.49, .9, "", transform=self.fig_title_axes.transAxes, fontsize=15, color='r', ha='center')
        if not self.simulated_data:
            self.add_legend()

    def initialize_accessories(self):

        if "motion_pattern" in self.accessories:
            self.nn_output_fig = plt.figure(figsize=(5, 5))
            self.nn_output_fig.suptitle('Network Output')
            self.kernel_fig = plt.figure(figsize=(5, 5))
            self.kernel_fig.suptitle('Motion pattern')

        if "velocities" in self.accessories:
            self.vel_fig = plt.figure(figsize=(12, 3))
            ma_ax = self.vel_fig.add_subplot(141)
            ma_plot = Plot(ma_ax, self.models[0].ma_vel, 1, False, False, False, colorbar_on=None, title=r'$P(V_{ma})$')
            vel_ax = self.vel_fig.add_subplot(142)
            vel_plot = Plot(vel_ax, self.models[0].ma_vel, 1, False, False, False, colorbar_on=None, title=r'$P(V_{pred})$')
            merge_vel_ax = self.vel_fig.add_subplot(143)
            merge_vel_plot = Plot(merge_vel_ax, self.models[0].ma_vel, 1, False, False, False, colorbar_on=None, title=r'$P(V_{merge})$')
            final_vel_ax = self.vel_fig.add_subplot(144)
            final_vel_plot = Plot(final_vel_ax, self.models[0].ma_vel, 1, False, False, False, colorbar_on=None, title='$P(V)$')
            self.accessories_plots = dict(ma_plot=ma_plot, vel_plot=vel_plot,
                                          merge_vel_plot=merge_vel_plot, final_vel_plot=final_vel_plot)

    def set_axis_ticks(self, extent):

        if self.vel_fig is not None:

            xlabels = (np.arange(extent) + np.array([-(extent // 2)])).tolist()
            ylabels = xlabels

            def format_fn_x(tick_val, tick_pos):
                if int(tick_val) in range(7):
                    return xlabels[int(tick_val)]
                else:
                    return ''

            def format_fn_y(tick_val, tick_pos):
                if int(tick_val) in range(7):
                    return ylabels[int(tick_val)]
                else:
                    return ''

            ax = self.vel_fig.get_axes()[0]
            max_extent = float(extent)
            ax.set_xticks(np.arange(.5, max_extent, 1.0))
            ax.set_yticks(np.arange(0.5, max_extent, 1.0))
            ax.xaxis.set_major_formatter(FuncFormatter(format_fn_x))
            ax.yaxis.set_major_formatter(FuncFormatter(format_fn_y))

            ylabel = ax.set_ylabel(r'$V_y$', color='darkred', fontsize=12)
            ylabel.set_rotation(0)
            ax.yaxis.set_label_coords(-0.06, .95)
            ax.set_xlabel(r'$V_x$', color='darkred', fontsize=12)
            ax.xaxis.set_label_coords(1.05, -0.025)

            for ax in self.vel_fig.get_axes()[1:]:
                ax.set_xticks([])
                ax.set_yticks([])

    def add_custom_element(self, plot):
        """Add extra elements to plot. This method has to be overwritten by subclasses. """
        pass

    def update_custom_element(self, idx):
        """ Update custom elements on animation. This method has to be overwritten by subclasses. """

    def initialize_models(self):
        """Initialize BOFUM models. This method has to be overwritten by subclasses."""
        pass

    def _draw_frame(self, framedata):

        t = self.models[0].t
        t_count = "frame = " + str(t)
        print(t_count)
        self.fig_title.set_text(t_count)

        measurement = self.models[0].measurement_at()
        for model in self.models:
            model.tracking_step(measurement=measurement)

        # plot new occupancy
        Ot_max = max(map(lambda model: model.P_Ot.max(), self.models))
        Ot_min = min(map(lambda model: model.P_Ot.min(), self.models))
        for i, model in enumerate(self.models):
            self.plots[i].set_axes_data("occupancy_axes", model.P_Ot, Ot_min, Ot_max)
            if self.show_seen:
                seen = model.evaluate_loc_at(t)
                self.plots[i].set_axes_data("seen_axes", seen)
            if self.plots[i].show_text:
                x_ent = model.calc_cross_entropy()
                f1_score = model.calc_f1_score()
                average_precision = model.calc_average_precision()
                self.plots[i].text.set_text("x_ent: {:.3f}, f1: {:.3f}, ap: {:.3f}".format(x_ent, f1_score, average_precision))
            self.update_custom_element(i)
            # if i == self.num_models-1:
            #     self.add_legend()
            self.plots[i].refresh_colorbar()

        # repeat tracking
        if framedata == self.num_steps-1:
            for model in self.models:
                model.reset()

    def new_frame_seq(self):
        return iter(range(self.num_steps))

    def _init_draw(self):
        pass

class TrackingAnimSimulation(TrackingAnimation):

    def __init__(self, models, num_steps, num_targets=1, diagonal=False, plot_map=True, **kwargs):

        self.num_targets = num_targets
        self.diagonal = diagonal
        self.trajs = None
        self.distances = None
        super(TrackingAnimSimulation, self).__init__(models, num_steps, True, plot_map=plot_map, **kwargs)

    def add_custom_element(self, plot):
        plot.add_traj_line(self.num_targets)

    def update_custom_element(self, idx):
        # add trajectory lines
        truncated_trajs = self.models[0].traversed_traj_at()

        for idx_, line in enumerate(self.plots[idx].lines):
            xs, ys = truncated_trajs[idx_].T[0][-5:], truncated_trajs[idx_].T[1][-5:]
            line.set_data(xs, ys)

    def initialize_models(self):
        self.distances , self.trajs = self.models[0].initialize(self.num_targets, self.num_steps)
        init_model = lambda model: model.initialize(self.num_targets, self.num_steps,
            distances=self.distances, trajectories=self.trajs)
        map(init_model, self.models[1:])

class TrackingAnimRealdata(TrackingAnimation):

    def __init__(self, models, num_steps, scene, plot_map=True,plot_seen=False, simulated_scenes=False, **kwargs):

        self.scene = scene
        self.simulated_scenes = simulated_scenes
        super(TrackingAnimRealdata, self).__init__(models, num_steps, False,
                                                     plot_seen=plot_seen, plot_map=plot_map, **kwargs)

    def update(self, scene, update_num_steps=True):
        self.scene = scene
        update_map = lambda plot: plot.set_axes_data("map_axes", self.scene.static_map)
        map(update_map, self.plots)
        map(lambda model: model.update(scene), self.models)
        if update_num_steps:
            self.num_steps = len(scene.hits)
        self.frame_seq = self.new_frame_seq()

    def initialize_models(self):
        init_model = lambda model: model.initialize(self.scene, not self.simulated_scenes)
        map(init_model, self.models)

    def add_custom_element(self, plot):
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
        b_patch = mpatches.Patch(color='b', label='False negative')
        o_patch = mpatches.Patch(color='orange', label='False positive')
        plt.legend(handles=[g_patch, b_patch, o_patch], bbox_to_anchor=(1, 1),
           bbox_transform=self.fig.transFigure)

    def update_custom_element(self, idx):
        t = self.models[0].t - 1
        model = self.models[idx]
        plot = self.plots[idx]
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
        # only show for occupancies higher than 1/2 highest occupancy
        h_max = false_positive.max() / 4
        false_positive = np.where(false_positive>h_max, false_positive, 0)


        Ot_max = max(map(lambda model: model.P_Ot.max(), self.models))
        plot.set_axes_data("fn_axes", false_negative, 0, 1)
        plot.set_axes_data("tp_axes", true_positive, 0, Ot_max)
        plot.set_axes_data("fp_axes", false_positive, 0, Ot_max)
        plot.set_axes_data("occupancy_axes", np.zeros_like(occupancy_prob))





