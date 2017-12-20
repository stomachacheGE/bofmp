
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

def remove_axis(xy):
    if xy == 'x':
        plt.xlabel('')
        plt.tick_params(axis='x', bottom='off', labelbottom='off')
    elif xy == 'y':
        plt.ylabel('')
        plt.tick_params(axis='y', left='off', labelleft='off')


def axis_adjustments(row, n_rows, column, title):
    if row == 0:
        plt.title(title)
    if row != n_rows - 1:
        remove_axis('x')
    if column != 0:
        remove_axis('y')

def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(map( reduced_cmap, step_list))
    new_LUT = np.array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

def plot_4d_tensor(arr, title=None, fig=None, **kwargs):
    if fig is not None:
        axarr = fig.subplots(*arr.shape[:2], sharex=True, sharey=True)
    else:
        fig, axarr = plt.subplots(*arr.shape[:2], sharex=True, sharey=True)
    arr = np.rot90(arr, 1, (0, 1))
    arr = np.rot90(arr, 1, (2, 3))

    xlabels = (np.arange(arr.shape[0]) + np.array([-(arr.shape[0]//2)])).tolist()
    ylabels = xlabels[::-1]

    def format_fn_x(tick_val, tick_pos):
        if int(tick_val) in range(arr.shape[0]):
            return xlabels[int(tick_val)]
        else:
            return ''

    def format_fn_y(tick_val, tick_pos):
        if int(tick_val) in range(arr.shape[0]):
            return ylabels[int(tick_val)]
        else:
            return ''

    for row_ix, row in enumerate(arr):
        for col_ix, arr_2d in enumerate(row):
            # axes = axarr[col_ix, row_ix].imshow(arr_2d.T,
            #                              vmin=arr.min(), vmax=arr.max(),
            #                              aspect='auto', **kwargs)


            axes = axarr[row_ix, col_ix].imshow(arr_2d,
                                         vmin=arr.min(), vmax=arr.max(),
                                         aspect='auto', **kwargs)
            axarr[row_ix, col_ix].set_axis_off()


    axarr[-1, 0].set_axis_on()

    axarr[-1, 0].xaxis.set_major_formatter(FuncFormatter(format_fn_x))
    axarr[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axarr[-1, 0].yaxis.set_major_formatter(FuncFormatter(format_fn_y))
    axarr[-1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

    ylabel = axarr[-1, 0].set_ylabel(r'$V_y$', color='darkred', fontsize=9)
    ylabel.set_rotation(0)
    axarr[-1, 0].yaxis.set_label_coords(-0.06, .95)
    axarr[-1, 0].set_xlabel(r'$V_x$', color='darkred', fontsize=9)
    axarr[-1, 0].xaxis.set_label_coords(1.05, -0.025)

    fig.text(.85, .05, r'$V_x^-$', fontdict=font)
    fig.text(.05, .88, r'$V_y^-$', fontdict=font)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axes, cax=cbar_ax)
    if title is not None:
        fig.suptitle(title)

def plot_4d_tensor_1(arr, title=None, **kwargs):

    fig, axarr = plt.subplots(*arr.shape[:2], sharex=True, sharey=True)
    for row_ix, row in enumerate(arr):
        for col_ix, arr_2d in enumerate(row):
            axes = axarr[row_ix, col_ix].imshow(arr_2d,
                                         vmin=arr.min(), vmax=arr.max(),
                                         aspect='auto', **kwargs)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axes, cax=cbar_ax)
    if title is not None:
        plt.title(title)