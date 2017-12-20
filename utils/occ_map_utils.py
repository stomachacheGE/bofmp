
import os
import ast
from copy import copy

import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap, Normalize, colorConverter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib import cm

from utils.plot_utils import remove_axis, axis_adjustments, cmap_map

# https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
# make the colormaps
black_cm = mpl.colors.LinearSegmentedColormap.from_list('black_cm',['white','black'], 2)
green_cm = mpl.colors.LinearSegmentedColormap.from_list('green_cm',['white','green'], 2)
blue_cm = mpl.colors.LinearSegmentedColormap.from_list('blue_cm',['white','blue'], 2)
red_cm = mpl.cm.get_cmap('OrRd')
greens_cm = mpl.cm.get_cmap('Greens')
greys_cm = mpl.cm.get_cmap('Greys')



black_cm._init() # create the _lut array, with rgba values
green_cm._init()
blue_cm._init()
red_cm._init()
greens_cm._init()
greys_cm._init()

# only show black and green color (alpha=0.6)
#black_cm._lut[:2, -1] = [0, 0.5]
green_cm._lut[:2, -1] = [0, 0.1]
blue_cm._lut[:2, -1] = [0, 0.7]
black_cm._lut[0, -1] = 0
# red colormap does not show white color
red_cm._lut[0, -1] = 0
greens_cm._lut[0, -1] = 0
greys_cm._lut[0, -1] = 0


def load_map(fname):
    yaml_file = fname[:-4] + '.yaml'
    if os.path.isfile(yaml_file):
        origin = \
            np.array(ast.literal_eval(parse_map_yaml(yaml_file, 'origin'))[:2])
        resolution = float(parse_map_yaml(yaml_file, 'resolution'))
    else:
        origin = None
        resolution = None
    # read image
    map_ = plt.imread(fname)
    if map_.ndim == 3:
        map_ = map_[..., 0].astype(bool)
    # invert image because walls are black and have an occupancy probability
    # of 1.0
    return np.logical_not(np.rot90(map_, k=-1)), origin, resolution


def save_map(fname, a):
    plt.imsave(fname, np.rot90(a), cmap='gray_r', vmin=0.0, vmax=1.0)


def free_space(map_):
    return np.logical_not(map_)


def parse_map_yaml(file, search_str):
    with open(file) as input_data:
        # Skips text before the beginning of the interesting block:
        for line in input_data:
            key, value = line.split(': ')
            value = value.rstrip('\n')
            if key == search_str:
                return value


def show_map(a, resolution=1.0, origin=None, cmap='gray_r', **kwargs):
    # rotate array back for showing
    a = np.rot90(a)
    if a.dtype == 'bool':
        # make map array transparent: add RGB layers in front
        rgb = np.zeros(np.hstack([a.shape, [3]]))
        a = np.concatenate([rgb, a[..., np.newaxis]], axis=-1)

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 10

    if 'extent' in kwargs:
        extent = kwargs['extent']
        kwargs.pop('extent')
    else:
        # origin is bottom left
        if origin is None:
            origin = np.array([0.0] * 2)
        height, width_ = np.array(a.shape[:2]) * resolution
        extent = np.array([origin[0], origin[0] + width_,
                           origin[1], origin[1] + height])
    if 'ax' in kwargs:
        ax = kwargs['ax']
        del kwargs['ax']
        axes = ax.imshow(a, cmap=cmap, interpolation='none', extent=extent, **kwargs)
    else:
        axes = \
            plt.imshow(a, cmap=cmap, interpolation='none', extent=extent, **kwargs)


    return axes

def show_traj(trajs):

    colors = cm.Dark2(np.linspace(0, 1, len(trajs)))
    for i, traj in enumerate(trajs):
        coordinate = traj.T.tolist()
        # starting point
        plt.plot(coordinate[0][0], coordinate[1][0], 'o', color=colors[i], zorder=13)
        plt.plot(coordinate[0], coordinate[1], color=colors[i], zorder=12)



def plot_occ_map(occ_map_arr, static_map=None, mask=None,
                 occ_map_res=None, map_res=1.0,
                 occ_map_origin=None, map_origin=None, colors=None,
                 vmin=0, vmax=None, y_label=None):
    # if just one occupancy map is given, add dimension
    if occ_map_arr.ndim == 2:
        occ_map_arr = copy(occ_map_arr[None, ...])

    n_agents = occ_map_arr.shape[0]

    if static_map is None:
        static_map = np.full_like(occ_map_arr[0], False, dtype=bool)
    if mask is None:
        mask = free_space(static_map).astype(float)
    if vmax == None:
        if occ_map_arr.shape == static_map.shape:
            vmax = occ_map_arr[~static_map].max()
        else:
            vmax = occ_map_arr.max()

    # if no different resolution for map is given, use the same as for cost map
    if occ_map_res is None:
        occ_map_res = map_res
    if occ_map_origin is None:
        occ_map_origin = map_origin



    # calculate image
    if colors is None:
        cms = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges]
        colors = np.array([cmap(256)[:3] for cmap in cms])
    im = np.ones(np.hstack([occ_map_arr.shape[-2:], 3]))
    for agent, occ_map in enumerate(occ_map_arr):
        im *= 1 - occ_map_arr.astype(float)[agent][..., None] / vmax * \
                  (1 - np.array(colors[agent])[None, None, :])

    im[im < 0] = 0

    # plot cost map
    cost_map_axis = show_map(im, cmap='OrRd', resolution=occ_map_res,
            vmin=vmin, vmax=vmax, origin=occ_map_origin)

    # create colormap list for output
    N = 256
    cmaps = [1 - np.linspace(0, 1, N)[:, None] * (1 - color[None, :]) for
             color in colors]
    cmaps = [mpl.colors.ListedColormap(cmap) for cmap in cmaps]


    # overlay mask
    # make alpha colormap
    N = 256
    cmap = np.tile(0.8, (N, 4))
    cmap[::-1, -1] = np.linspace(0, 1, N)
    cmap = ListedColormap(cmap)
    mask_axes = show_map(mask, resolution=map_res, origin=map_origin,
                         cmap=cmap, vmin=0.0, vmax=1.0)
    # overlay walls
    map_axis = show_map(static_map, resolution=map_res, origin=map_origin)
    # add axis labels
    plt.xlabel("x (m)")
    y_label_ = "y (m)"
    if y_label is not None:
        y_label_ = y_label + '\n' + y_label_
    plt.ylabel(y_label_)
    return cost_map_axis, map_axis, cmaps

def display_occ_map(occ_map, static_map=None, mask=None, cost_map_res=1.0,
                    map_res=None, cost_map_origin=(0, 0), map_origin=None,
                    **kwargs):
    if np.ndim(occ_map) == 2:
        occ_map = occ_map[None, ...]
    return display_occ_maps(occ_map[None, None, :], static_map=static_map)


def display_data_set(pred, kl_div, label, maps, masks, model_name='', diff_label=''):
    # todo: use display_occ_maps function

    rows_per_plot = 3
    aug_fac = 8
    samples_per_figure = aug_fac * rows_per_plot
    num_figs = int(maps.shape[0] / samples_per_figure)

    for fig in range(num_figs):
        sample_ixs = fig * samples_per_figure + \
                     np.arange(0, samples_per_figure, aug_fac, dtype=int)
        resolution = 0.2
        sample_data = \
            np.array([pred[sample_ixs], kl_div[sample_ixs], label[sample_ixs]])
        sample_data = sample_data.swapaxes(0, 1)
        titles = [model_name + ' prediction', diff_label, 'ground truth']
        occ_map_vmax = sample_data[:, [0, 2]].max()
        vmaxs = [occ_map_vmax, kl_div.max(), occ_map_vmax]
        texts = np.full([rows_per_plot, 3], '    ')
        for row in range(rows_per_plot):
            texts[row, 1] = '{:.3f}'.format(kl_div[sample_ixs][row].sum())
        display_occ_maps(sample_data, maps[sample_ixs], masks[sample_ixs],
                         resolution, titles=titles, vmaxs=vmaxs, labels=texts)
        plt.show()

def display_trans_map(transition_map, map_arr, last, next, order,
                      map_res=None, grey_out=None, cmap='OrRd', **kwargs):
    map_ = transition_map[:, :, last, next]
    # map_ /= np.sum(map_)
    # print(len(np.array(np.where(map_ != 0.25)).T.tolist()))
    fig, axes = display_occ_map(map_, map_arr, order=order, map_res=map_res,
                                cmap=cmap,
                                grey_out=grey_out, **kwargs)
    directions = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    fig.set_title('P(next={}|last={})'.format(directions[next], directions[last]),
                  fontsize=10)

    return axes


def display_data_set(pred, kl_div, label, maps, masks, model_name='', diff_label=''):
    rows_per_plot = 3
    aug_fac = 8
    samples_per_figure = aug_fac * rows_per_plot
    num_figs = int(maps.shape[0] / samples_per_figure)

    for fig in range(num_figs):
        sample_ixs = fig * samples_per_figure + \
                     np.arange(0, samples_per_figure, aug_fac, dtype=int)
        resolution = 0.2
        sample_data = \
            np.array([pred[sample_ixs], kl_div[sample_ixs], label[sample_ixs]])
        sample_data = sample_data.swapaxes(0, 1)
        titles = [model_name + ' prediction', diff_label, 'ground truth']
        occ_map_vmax = sample_data[:, [0, 2]].max()
        vmaxs = [occ_map_vmax, kl_div.max(), occ_map_vmax]
        texts = np.full([rows_per_plot, 3], '    ')
        for row in range(rows_per_plot):
            texts[row, 1] = '{:.3f}'.format(kl_div[sample_ixs][row].sum())
        display_occ_maps(sample_data, maps[sample_ixs], masks[sample_ixs],
                         resolution, titles=titles, vmaxs=vmaxs, labels=texts)
        plt.show()

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def plot_grid_map_hmm(transitions, mode, grid_res=1.0, grid_origin=None,
                      map_=None, map_origin=None, map_res=1.0):
    """ Plot quiver plots which indicates traffic on different directions.
        Mode has to be either 'counts' or 'probs'.
    """

    if grid_origin is None:
        grid_origin = [0.0, 0.0]

    if mode == 'counts':
        up, right, down, left = get_quiver_from_counts(transitions)
        #TODO: should not normalize over all values, should normalize over "to" dimension
        # we need to normalize our colors array to match it colormap domain
        # which is [0, 1]
        values = np.array([up, right, down, left])
        norm = Normalize()
        norm.autoscale(values)
        values = norm(values)
    elif mode == 'probs':
        up, right, down, left = get_quiver_from_probs(transitions)
        values = np.array([up, right, down, left])
    else:
        return

    # calculate x ans y axis
    x, y = \
        [(np.arange(up.shape[ix]) + 0.5) * grid_res + \
         grid_origin[ix] for ix in [0, 1]]
    x, y = np.meshgrid(x, y)
    quiveropts = \
        dict( width=0.005, scale=1/0.15, headaxislength=0, headlength=0, zorder=9)

    # colormap = cm.magma
    # pick your colormap here, refer to
    # http://matplotlib.org/examples/color/colormaps_reference.html
    # and
    # http://matplotlib.org/users/colormaps.html
    # for details
    plt.rcParams['image.cmap'] = 'Greens'
    # plot up down left right lines
    # right
    #plt.quiver(x, y, np.ones_like(right) * 0.1, np.zeros_like(right), color=colormap(values[0, ...]), linewidths=np.digitize(right.flatten(), bins)*3, **quiveropts)
    plt.quiver(x, y, np.ones_like(right) * 0.1, np.zeros_like(right), values[1, ...], **quiveropts)
    # left
    plt.quiver(x, y, -np.ones_like(left)* 0.1, np.zeros_like(left), values[3,...], **quiveropts)
    # # up
    plt.quiver(x, y, np.zeros_like(up), np.ones_like(up)* 0.1, values[0,...], **quiveropts)
    # # down
    axes = plt.quiver(x, y, np.zeros_like(down), -np.ones_like(down)* 0.1, values[2,...], **quiveropts)

    # plot map
    if map_ is not None:
        show_map(map_, resolution=map_res, origin=map_origin)

    return axes

def get_quiver_from_counts(transition_counts):
    # transition counts has shape of [width, height, from_direction, to_direction]
    from_to_counts = np.array([np.sum(transition_counts, axis=ix)
                               for ix in [3, 2]])
    with np.errstate(divide='ignore', invalid='ignore'):
        from_to_probs = \
            from_to_counts / np.sum(from_to_counts, axis=-1, keepdims=True)
        from_to_probs[np.isnan(from_to_probs)] = 0
    from_to_probs = np.swapaxes(from_to_probs, 1, 2)

    # calculate line lengths
    up = (from_to_probs[0, ..., DOWN] + from_to_probs[1, ..., UP]) / 2
    down = (from_to_probs[0, ..., UP] + from_to_probs[1, ..., DOWN]) / 2
    right = (from_to_probs[0, ..., LEFT] + from_to_probs[1, ..., RIGHT]) / 2
    left = (from_to_probs[0, ..., RIGHT] + from_to_probs[1, ..., LEFT]) / 2

    return up, right, down, left

def get_quiver_from_probs(transition_probs):
    # transition probs has shape of [from_direction, to_direction, width, height]
    to_probs = np.sum(transition_probs, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        to_probs = to_probs / np.sum(to_probs, axis=0, keepdims=True)
        to_probs[np.isnan(to_probs)] = 0
    to_probs = np.swapaxes(to_probs, 1, 2)

    up = to_probs[UP, ...]
    down = to_probs[DOWN, ...]
    right = to_probs[RIGHT, ...]
    left = to_probs[LEFT, ...]
    return up, right, down, left


def display_occ_maps(occ_maps, static_map, resolution=1.0,
                     c_labels=None, col_labels=None, row_labels=None,
                     map_origin=None):
    if np.ndim(occ_maps) == 4:
        occ_maps = occ_maps[None, ...]

    n_rows, n_columns, n_agents = occ_maps.shape[:3]
    map_width, map_height = occ_maps.shape[-2:]
    width_ = n_columns * 6 + 11
    height_ = n_rows * map_height / map_width * 2.0 + 8
    width = 16
    height = height_ / width_ * width
    plt.figure(figsize=(width, height), dpi=80)

    # define geometry of figure
    left, right = [0.7 / width, 1 - 0.7 / width]
    bottom, top = [0.7 / height, 1 - 0.7 / height]
    wspace, hspace = [0.2 / width, 0.2 / height]

    # plot main layout
    gs1 = gridspec.GridSpec(n_rows, n_columns)
    gs1_wspace = 0.05
    gs1.update(left=left, right=right,
               bottom=bottom, top=top,
               wspace=gs1_wspace, hspace=hspace)

    c_bar_width = 0.15 * n_agents / width
    c_bar_left = right - c_bar_width
    # add colorbar layout
    gs1.update(right=c_bar_left - wspace)

    gs2_wspace = 0.5
    gs2 = gridspec.GridSpec(n_rows, n_agents)
    gs2.update(left=c_bar_left, right=right,
               bottom=bottom, top=top,
               wspace=gs2_wspace, hspace=hspace)

    for row in range(n_rows):
        # default settings
        remove_x_axis = True
        col_labels_row = None
        if row == 0:
            col_labels_row = col_labels
        if row == n_rows - 1:
            remove_x_axis = False

        row_label = None
        if row_labels is not None:
            row_label = row_labels[row]

        c_label = None
        if c_labels is not None:
            c_label = c_labels[row]

        plot = plot_row(occ_maps[row], static_map, resolution, gs1, gs2, row,
                        remove_x_axis=remove_x_axis, col_labels=col_labels_row,
                        row_label=row_label, c_label=c_label, map_origin=map_origin)

    return plot



def plot_row(occ_maps, static_map, res, plot_gs, cmap_gs, row_ix, vmax=None,
             remove_x_axis=True, c_label='agent occupancy', row_label=None,
             col_labels = None, map_origin=None):
    """
    occ_maps: n_columns x n_agents x width x height array of occupancy maps
    gs: matplotlib grid spec
    vmax: colormap maximum value
    """
    n_cols, n_agents, _, _ = occ_maps.shape
    vmax = vmax or occ_maps.max()
    for col in range(n_cols):
        row_label_col = None
        if col == 0:
            row_label_col = row_label
        plt.subplot(plot_gs[row_ix, col])
        img1, _, cmaps = plot_occ_map(occ_maps[col], static_map,
                                      map_res=res,
                                      vmax=vmax, y_label=row_label,
                                      map_origin=map_origin,
                                      occ_map_origin=map_origin)
        # remove axis where necessary
        if remove_x_axis:
            remove_axis('x')
        if col != 0:
            remove_axis('y')

        # add title
        if col_labels is not None:
            plt.title(col_labels[col])

    # add colorbar in last column
    bounds = np.linspace(0, vmax, 251)
    ticks = np.linspace(0, 1, 5) * np.round(
        vmax, int(-np.floor(np.log10(vmax))))
    for agent in range(n_agents):
        cax = plt.subplot(cmap_gs[row_ix, agent])
        c_bar = mpl.colorbar.ColorbarBase(cax, cmap=cmaps[agent],
                                          boundaries=bounds,
                                          ticks=ticks)
        # add label to last colorbar
        if agent == n_agents - 1:
            c_bar.set_label(c_label)
        else:
            c_bar.set_ticklabels([])
    return img1



def display_random_nn_io(inputs, labels, masks=None, num_samples=100, \
                                                            resolution=1.0):
    if masks is None:
        masks = [None] * len(inputs)
    # plot some row inputs / labels
    for _ in range(num_samples):
        ix = np.random.randint(0, len(inputs))
        row_input = inputs[ix]
        row_label = labels[ix]
        row_mask = masks[ix]
        display_occ_map(row_label, row_input, row_mask, resolution)
        plt.show()


def plot_trajectories(trajs, map_arr, map_res=1.0):
    trajs = (trajs + np.array([0.5, 0.5])) * map_res
    for trajectory in trajs:
        plt.plot(trajectory[:, 0], trajectory[:, 1])
    show_map(map_arr, resolution=map_res)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


