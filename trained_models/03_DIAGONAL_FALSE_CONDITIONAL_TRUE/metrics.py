import theano.tensor as T
import numpy as np

from lasagne.objectives import binary_crossentropy, categorical_crossentropy

def theano_metrics(y_pred, y_true, n_classes, void_labels):
    """
    Returns the intersection I and union U (to compute the jaccard I/U) and the accuracy.

    :param y_pred: tensor of predictions. shape  (b*0*1, c) with c = n_classes
    :param y_true: groundtruth, shape  (b,0,1) or (b,c,0,1) with c=1
    :param n_classes: int
    :param void_labels: list of indexes of void labels
    :return: return tensors I and U of size (n_classes), and scalar acc
    """

    # Put y_pred and y_true under the same shape
    y_true = T.flatten(y_true)
    y_pred = T.argmax(y_pred, axis=1)

    # We use not_void in case the prediction falls in the void class of the groundtruth
    for i in range(len(void_labels)):
        if i == 0:
            not_void = T.neq(y_true, void_labels[i])
        else:
            not_void = not_void * T.neq(y_true, void_labels[i])

    I = T.zeros(n_classes)
    U = T.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = T.eq(y_true, i)
        y_pred_i = T.eq(y_pred, i)
        I = T.set_subtensor(I[i], T.sum(y_true_i * y_pred_i))
        U = T.set_subtensor(U[i], T.sum(T.or_(y_true_i, y_pred_i) * not_void))

    accuracy = T.sum(I) / T.sum(not_void)

    return I, U, accuracy


def numpy_metrics(y_pred, y_true, n_classes, void_labels):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.flatten()

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~ np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy


def crossentropy(y_pred, y_true, void_labels):
    # Flatten y_true
    y_true = T.flatten(y_true)
    
    # Clip predictions

    # Create mask
    mask = T.ones_like(y_true)
    for el in void_labels:
        mask = T.switch(T.eq(y_true, el), np.int32(0), mask)

    # Modify y_true temporarily
    y_true_tmp = y_true * mask

    # Compute cross-entropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true_tmp)

    # Compute masked mean loss
    loss *= mask
    loss = T.sum(loss) / T.sum(mask).astype('float32')

    return loss


def squared_error_void(y_pred, y_true):
    # Flatten y_true
    y_true = T.flatten(y_true)
    y_pred = T.flatten(y_pred)
    # Create mask
    mask = T.ones_like(y_true, dtype=np.int32)
    mask = T.switch(T.isclose(y_true, 1), np.int32(0), mask)

    # Modify y_true temporarily
    y_true_tmp = y_true * mask

    error = mask * T.sqr(y_true_tmp - y_pred)

    return T.mean(error)

def squared_error_void_np(y_pred, y_true):
    error = (y_true - y_pred)**2
    return np.mean(error[np.logical_not(np.isclose(y_true, 1.0))])


def binary_crossentropy_void(y_pred, y_true, y_mask):
    # Flatten y_true
    y_true = T.reshape(y_true, y_pred.shape)
    y_mask = T.reshape(y_mask, y_pred.shape)

    eps = 1e-12
    y_pred = y_pred.clip(0 + eps, 1 - eps)

    error = y_mask * binary_crossentropy(y_pred, y_true)

    return T.mean(error)

def binary_kl_void(y_pred, y_true, y_mask):
    ent = binary_crossentropy_void(y_true, y_true, y_mask)
    ce = binary_crossentropy_void(y_pred, y_true, y_mask)
    kl = ce - ent
    return kl

def binary_rev_kl_void(y_pred, y_true, y_mask):
    return binary_kl_void(y_true, y_pred, y_mask)

def binary_sym_kl_void(y_pred, y_true, y_mask):
    return (binary_kl_void(y_true, y_pred, y_mask) + binary_rev_kl_void(y_pred, y_true, y_mask)) / 2

def categorical_crossentropy_void(y_pred, y_true, y_mask, no_mask=False, conditional=False):
    # Flatten y_true
    # y_true = T.reshape(y_true, y_pred.shape)
    # y_mask = T.reshape(y_mask, y_pred.shape)

    # # Create mask
    # mask = T.ones_like(y_pred, dtype=np.int32)
    # mask = T.switch(y_pred >= 1, np.int32(0), mask)
    #
    # # Modify y_pred temporarily
    # y_pred_tmp = y_pred * mask + 0.99 * T.ones_like(y_pred) * (1 - mask)

    batch_size, n_channels, n_rows, n_cols = y_true.shape


    eps = 1e-12
    y_pred = y_pred.clip(0 + eps, 1 - eps)

    if not no_mask:
        loss = - y_mask * y_true * T.log(y_pred)
        loss = T.sum(loss)
        average_loss = loss / batch_size
    else:
        loss = - y_true * T.log(y_pred)
        loss = T.sum(loss)
        if conditional:
            n_directions = np.sqrt(n_channels).astype('int32')
            average_loss = loss / (batch_size * n_directions * n_cols * n_rows)
        else:
            average_loss = loss / (batch_size * n_cols * n_rows)
    return average_loss

def categorical_crossentropy_w_epsilon_void(y_pred, y_true, y_mask, no_mask=False):
    # Flatten y_true
    # y_true = T.reshape(y_true, y_pred.shape)
    # y_mask = T.reshape(y_mask, y_pred.shape)

    # # Create mask
    # mask = T.ones_like(y_pred, dtype=np.int32)
    # mask = T.switch(y_pred >= 1, np.int32(0), mask)
    #
    # # Modify y_pred temporarily
    # y_pred_tmp = y_pred * mask + 0.99 * T.ones_like(y_pred) * (1 - mask)

    batch_size, n_channels, n_rows, n_cols = y_true.shape
    n_directions = np.sqrt(n_channels).astype('int32')

    eps = 1e-12
    y_pred = y_pred.clip(0 + eps, 1 - eps)

    if not no_mask:
        loss = - y_mask * (y_true + eps) * T.log(y_pred)
        loss = T.sum(loss)
        average_loss = loss / batch_size
    else:
        loss = - (y_true + eps) *  T.log(y_pred)
        loss = T.sum(loss)
        average_loss = loss / (batch_size * n_directions * n_cols * n_rows)

    return average_loss

def mean_square_void(y_pred, y_true, y_mask, no_mask=False):

    batch_size, n_channels, n_rows, n_cols = y_true.shape
    n_directions = np.sqrt(n_channels).astype('int32')
    y_true = y_true.reshape((batch_size, n_directions, n_directions, n_rows, n_cols))
    y_pred = y_pred.reshape((batch_size, n_directions, n_directions, n_rows, n_cols))

    errors = T.sqrt(T.sum(T.square(y_true-y_pred), axis=2))
    return T.sum(errors)

def categorical_kl_void(y_pred, y_true, y_mask):
    ent = categorical_crossentropy_void(y_true, y_true, y_mask)
    ce = categorical_crossentropy_void(y_pred, y_true, y_mask)
    kl = ce - ent
    return kl

def categorical_reverse_kl_void(y_pred, y_true, y_mask):
    return categorical_kl_void(y_true, y_pred, y_mask)

def categorical_sym_kl_void(y_pred, y_true, y_mask):
    return (categorical_kl_void(y_true, y_pred, y_mask) + categorical_reverse_kl_void(y_pred, y_true, y_mask)) / 2

def earth_mover_distance_asym(y_pred, y_true, y_mask, axis_order='xy'):
    y_pred = T.reshape(y_pred, y_true.shape)
    y_pred *= y_mask
    y_true *= y_mask

    y_true = y_true / y_true.sum()
    y_pred = y_pred / y_pred.sum()

    if axis_order == 'yx':
        y_true = y_true.dimshuffle([0, 1, 3, 2])
        y_pred = y_pred.dimshuffle([0, 1, 3, 2])

    # calculate approximate earth mover distance to transform probability
    # distribution y_true into y_pred

    emd = 0.0

    # calculate how much probability mass has to be moved along rows, in x direction
    diff = y_pred - y_true
    move_x = diff.sum(axis=2, keepdims=True).cumsum(axis=3)[..., :, :-1]
    # calculate from which cells to take the probability mass
    move_x_weights = diff.cumsum(axis=3)[..., :, :-1]
    # use only positions where sign is right
    move_x_weights = T.set_subtensor(move_x_weights[T.neq(T.sgn(move_x_weights), T.sgn(move_x)).nonzero()], 0)
    # normalize weightings to one
    # set weights uniformely to one, if all are zero
    move_x_weights = T.set_subtensor(move_x_weights[T.eq(move_x_weights.sum(axis=2, keepdims=True), 0).nonzero()], 1)
    move_x_weights /= move_x_weights.sum(axis=2, keepdims=True)
    # apply weighting
    move_x = move_x * move_x_weights
    emd += np.abs(move_x).sum()

    y_true_trans = y_true
    y_true_trans += T.set_subtensor(y_true_trans[..., :, :-1], move_x)
    y_true_trans -= T.set_subtensor(y_true_trans[..., :, 1:], move_x)

    # move mass along columns, in y direction
    diff = y_pred - y_true_trans
    move_y = diff.cumsum(axis=2)[..., :-1, :]
    emd += np.abs(move_y).sum()

    # check if we get y_pred
    y_true_trans2 = y_true_trans
    y_true_trans2 += T.set_subtensor(y_true_trans2[..., :-1, :], move_y)
    y_true_trans2 -= T.set_subtensor(y_true_trans2[..., 1:, :], move_y)

    return emd


def earth_mover_distance_void(y_pred, y_true, y_mask):
    x = earth_mover_distance_asym(y_pred, y_true, y_mask, axis_order='xy')
    y = earth_mover_distance_asym(y_pred, y_true, y_mask, axis_order='yx')

    return (x + y) / 2

def cross_entropy(caculate_at, P_occ_obser, P_occ_pred):
    """ Caculate cross entropy between prediction and ground truth."""

    # only calculate for specified locations on map
    locs = np.where(caculate_at)
    P_occ_obser = P_occ_obser[locs]
    P_occ_pred = P_occ_pred[locs]

    # cliping for numeric stability
    eps = 1e-12
    P_occ_pred = P_occ_pred.clip(0 + eps, 1 - eps)

    P_n_occ_obser = 1 - P_occ_obser
    P_n_occ_pred = 1 - P_occ_pred
    res = - (P_occ_obser * np.log(P_occ_pred) + P_n_occ_obser * np.log(P_n_occ_pred))
    # print("res shape is {}".format(res.shape))
    return np.mean(res)