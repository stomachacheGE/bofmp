
import os
from copy import copy
import cPickle

import numpy as np
import scipy.stats

DAT_FOLDER = '/local/data/'
SIMPLE_MAP = 'simple_map'


def blur2int(blur, max_blur=np.inf):
    blur_int = int(2 * round(2 * blur) - 1)
    blur_int = min(blur_int, max_blur)
    return blur_int

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)



def random_search(function, save_file, num_pts, *arg_fns, **kwarg_fns):
    for ix in range(num_pts):
        print("random search: sampling point {}/{}".format(ix, num_pts))
        args = []
        kwargs = {}

        # get random parameters
        for arg_fn in arg_fns:
            args.append(arg_fn())

        for key in kwarg_fns:
            kwargs[key] = kwarg_fns[key]()

        print(args, kwargs)

        returns = function(*args, **kwargs)
        # convert returns to list
        if not isinstance(returns, tuple):
            returns = [returns]
        else:
            returns = list(returns)

        # save to file
        with open(save_file, 'a') as f_handle:
            f_handle.write(',\t'.join(map(str, args + list(kwargs.values()) +
                                          returns)) + '\n')

def binary_cross_entropy(p, q, mask, elementwise=False):
    # if p equals q, make sure 0*log(0) returns 0
    # mask_tmp = copy(mask)
    # if np.array_equal(p, q):
    #     mask_tmp *= q != 0
    #     mask_tmp *= q != 1
    # # make sure masked pixels dont make problems
    # p[mask_tmp == 0.0] = 0.5
    # q[mask_tmp == 0.0] = 0.5

    q[q >= 1] = 0.9999999
    q[q <= 0] = 0.0000001

    cross_ent = mask * (-p * np.log(q) - (1 - p) * np.log(1 - q))

    if elementwise:
        return cross_ent
    else:
        return cross_ent.sum()

def binary_kl(p, q, mask, elementwise=False):
    ent = binary_cross_entropy(p, p, mask, elementwise=elementwise)
    ce = binary_cross_entropy(p, q, mask, elementwise=elementwise)

    return ce - ent

def binary_rev_kl(p, q, mask, elementwise=False):
    return binary_kl(q, p, mask, elementwise=elementwise)

def binary_sym_kl(p, q, mask, elementwise=False):
    return (binary_kl(q, p, mask, elementwise=elementwise) + binary_rev_kl(p, q, mask, elementwise=elementwise)) / 2

def categorical_cross_entropy(p, q, mask, elementwise=False, no_mask=False, conditional=False):
    # if p equals q, make sure 0*log(0) returns 0
    # mask_tmp = copy(mask)
    # if np.array_equal(p, q):
    #     mask_tmp *= q != 0
    #     mask_tmp *= q != 1
    # # make sure masked pixels dont make problems
    # p[mask_tmp == 0.0] = 0.5
    # q[mask_tmp == 0.0] = 0.5

    batch_size, n_channels, n_rows, n_cols = p.shape


    q_tmp = copy(q)
    eps = 1e-12
    q_tmp[q >= 1] = 1 - eps
    q_tmp[q <= 0] = eps

    if not no_mask:
        cross_ent = mask * -p * np.log(q_tmp)
        cross_ent /=  batch_size
    else:
        cross_ent = -p * np.log(q_tmp)
        if conditional:
            n_directions = np.sqrt(n_channels)
            cross_ent /= batch_size * n_directions * n_cols * n_rows
        else:
            cross_ent /= batch_size * n_cols * n_rows

    if elementwise:
        return cross_ent
    else:
        return cross_ent.sum()


def categorical_kl_div(preds, labels, masks, elementwise=False, no_mask=False, conditional=False):
    ent = categorical_cross_entropy(labels, labels, masks, elementwise, no_mask, conditional)
    ce = categorical_cross_entropy(labels, preds, masks, elementwise, no_mask, conditional)
    kl = ce - ent

    return kl


def categorical_reverse_kl_div(preds, labels, masks, elementwise=False):
    rev_kl =  categorical_kl_div(labels, preds, masks, elementwise=elementwise)
    return rev_kl

def categorical_sym_kl_div(preds, labels, masks, elementwise=False):
    sym_kl =  (categorical_kl_div(preds, labels, masks, elementwise=elementwise)
            + categorical_reverse_kl_div(preds, labels, masks,
                                         elementwise=elementwise)) / 2
    return sym_kl


def diff(p, q, mask):
    diff = mask * (p - q)
    return diff

def get_file(fname, type):
    tags = fname.split('/')
    home_lvl = len(DAT_FOLDER.split('/')) - 2
    tags = tags[home_lvl + 1:]

    # interpret file tree until map string as data type
    maps = [f for f in os.listdir(os.path.join(DAT_FOLDER, 'maps'))]
    while tags[1] not in maps:
        tags[0] = os.path.join(*tags[:2])
        tags.pop(1)

    tags[0] = type

    occ_map_files = ['hit', 'seen']

    # interpret file right if it has suffix
    if tags[-1].split('_')[-1].split('.')[0] in occ_map_files:
        tags[-1] = tags[-1].split('_')[0]
    else:
        tags[-1] = tags[-1].split('.')[0]


    ftype = 'npy'
    if type == 'maps':
        ftype = 'png'
        tags[2] = 'thresholded_20'
        tags = tags[:3]
    elif type.startswith('occ_maps'):
        return [os.path.join(DAT_FOLDER, *tags) + '_' + arr_str +
                '.' + ftype for arr_str in occ_map_files]
    return os.path.join(DAT_FOLDER, *tags) + '.' + ftype


def get_npy(files, function, *args, **kwargs):
    if not isinstance(files, list):
        files_ = [files]
    else:
        files_ = files

    recalc = False
    for file in files_:
        if not os.path.isfile(file):
            recalc = True
    if recalc:
        ret_arr = function(*args, **kwargs)
        if not isinstance(files, list):
            ret_arr = [ret_arr]
        for i, file in enumerate(files_):
            ensure_dir(file)
            np.save(file, ret_arr[i])
    ret_arr = []
    for file in files_:
        ret_arr.append(np.load(file, encoding='latin1'))

    if not isinstance(files, list):
        return ret_arr[0]
    else:
        return ret_arr


def unique_rows(a):
    unique_a = a[0]
    for ix in range(1, a.shape[0]):
        if not (a[ix - 1] == a[ix]).all():
            unique_a = np.vstack([unique_a, a[ix]])
    return unique_a

# -------------- deprecated ----------------- #

# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


# def square_gaussian(size, fwhm=3, center=None):
#     """ Make a square gaussian kernel.
#     size is the length of a side of the square
#     fwhm is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """
#
#     x = np.arange(0, size, 1, float)
#     y = x[:, np.newaxis]
#
#     if center is None:
#         x0 = y0 = size // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]
#
#     return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def nd_gaussian(shape, center, cov):
    """ Make a square gaussian kernel.
    shape is the shape of the output array
    cov is the covariance (we assume a diagonal covariance matrix)
    """

    edge_grids = np.meshgrid(*[np.arange(-0.5, i + 0.5, 1.0, float)
                               for i in shape], indexing='ij')
    end_slices = [slice(1, i + 1) for i in shape]
    start_slices = [slice(0, i) for i in shape]

    cum = np.ones(shape)
    for i, edge_grid in enumerate(edge_grids):
        cum_i = scipy.stats.norm.cdf(edge_grid, loc=center[i], scale = cov)
        cum *= cum_i[end_slices] - cum_i[start_slices]

    return cum

def pickle_save(file, obj):
    with open(file, 'wb') as f:
        cPickle.dump(obj, f)

def pickle_load(file):
    with open(file, 'rb') as f:
        obj = cPickle.load(f)
    return obj