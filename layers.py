from lasagne.layers import (
    NonlinearityLayer, Conv2DLayer, DropoutLayer, Pool2DLayer, ConcatLayer, Deconv2DLayer,
    DimshuffleLayer, ReshapeLayer, get_output, BatchNormLayer, get_all_param_values, Layer)

from lasagne.nonlinearities import linear, softmax, sigmoid
from lasagne.init import HeUniform

import numpy as np
import theano.tensor as T

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2, pad='same'):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    """

    l = NonlinearityLayer(BatchNormLayer(inputs))
    l = Conv2DLayer(l, n_filters, filter_size, pad=pad, W=HeUniform(gain='relu'),
                    nonlinearity=linear,
                    flip_filters=False)
    if dropout_p != 0.0:
        l = DropoutLayer(l, dropout_p)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = Pool2DLayer(l, 2, mode='max')

    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = ConcatLayer(block_to_upsample)
    l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
    # Concatenate with skip connection
    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution

def TransitionUpRes(skip_connection, block_to_upsample, n_filters_keep, dropout_p=0.2):
    l = TransitionUp(skip_connection, [block_to_upsample],
                     block_to_upsample.output_shape[1])
    l = BN_ReLU_Conv(l, n_filters_keep, filter_size=1, dropout_p=dropout_p)


    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution

def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2DLayer(inputs, n_classes, filter_size=1, nonlinearity=linear, W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax

    l = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = get_output(l).shape
    l = ReshapeLayer(l, (batch_size * n_rows * n_cols, n_classes))
    l = NonlinearityLayer(l, softmax)
    l = ReshapeLayer(l, (batch_size, n_rows, n_cols, n_classes))
    l = DimshuffleLayer(l, (0, 3, 1, 2))
    l = ReshapeLayer(l, (batch_size, n_classes, n_rows, n_cols))

    return l

    # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
    # any improvements. Our guess is that up_cnn naturally permits this multiscale approach


def SigmoidLayer(inputs):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2DLayer(inputs, 1, filter_size=1, nonlinearity=linear,
                    W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax

    l = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = get_output(l).shape
    l = ReshapeLayer(l, (batch_size * n_rows * n_cols, 1))
    l = NonlinearityLayer(l, sigmoid)

    return l

    # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
    # any improvements. Our guess is that up_cnn naturally permits this multiscale approach


def SpatialSoftmaxLayer(inputs):
    """
    Performs 1x1 convolution followed by spatial softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2DLayer(inputs, 1, filter_size=1, nonlinearity=linear,
                    W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, 1, n_rows, n_cols) to
    #        (batch_size, n_rows * n_cols, 1, 1)
    #     2. Apply softmax

    l = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = get_output(l).shape
    l = ReshapeLayer(l, (batch_size, n_rows * n_cols))
    l = NonlinearityLayer(l, softmax)
    l = ReshapeLayer(l, (batch_size, 1, n_rows, n_cols))

    return l

    # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
    # any improvements. Our guess is that up_cnn naturally permits this multiscale approach

def TransitionalSoftmaxLayer(inputs, n_directions):
    """
    Performs 1x1 convolution followed by softmax nonlinearity.
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2DLayer(inputs, n_directions**2, filter_size=1, nonlinearity=linear, W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax

    batch_size, n_channels, n_rows, n_cols = get_output(l).shape
    l = ReshapeLayer(l, (batch_size, n_directions, n_directions, n_rows, n_cols))
    l = DimshuffleLayer(l, (0, 1, 3, 4, 2))
    l = ReshapeLayer(l, (batch_size * n_directions * n_rows * n_cols, n_directions))
    l = NonlinearityLayer(l, softmax)
    l = ReshapeLayer(l, (batch_size, n_directions, n_rows, n_cols, n_directions))
    l = DimshuffleLayer(l, (0, 1, 4, 2, 3))
    l = ReshapeLayer(l, (batch_size, n_channels, n_rows, n_cols))

    return l

def TransitionalNormalizeLayer(inputs, n_directions):
    """
    Performs 1x1 convolution followed by softmax nonlinearity.
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2DLayer(inputs, n_directions**2, filter_size=1, nonlinearity=linear, W=HeUniform(gain='relu'), pad='same',
                    flip_filters=False, stride=1)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax

    batch_size, n_channels, n_rows, n_cols = get_output(l).shape
    l = ReshapeLayer(l, (batch_size, n_directions, n_directions, n_rows, n_cols))
    l = DimshuffleLayer(l, (0, 1, 3, 4, 2))
    l = ReshapeLayer(l, (batch_size * n_directions * n_rows * n_cols, n_directions))
    l = NormalizeLayer(l)
    l = ReshapeLayer(l, (batch_size, n_directions, n_rows, n_cols, n_directions))
    l = DimshuffleLayer(l, (0, 1, 4, 2, 3))
    l = ReshapeLayer(l, (batch_size, n_channels, n_rows, n_cols))

    return l

class NormalizeLayer(Layer):
    def get_output_for(self, input, **kwargs):
        # batch_size, n_channels, n_rows, n_cols = self.input_shape
        input = input - input.min(axis=1, keepdims=True)
        output = input / input.sum(axis=1, keepdims=True)
        # deal with NaN produced because of dividing by 0
        nan_mask = T.isnan(output)
        nan_idx = nan_mask.nonzero()
        output_without_nan = T.set_subtensor(output[nan_idx], 0)
        inf_mask = T.isinf(output_without_nan)
        inf_idx = inf_mask.nonzero()
        output_without_inf = T.set_subtensor(output_without_nan[inf_idx], 0)
        return output_without_inf

    # def get_output_shape_for(self, input_shape):
    #     batch_size, n_channels, n_rows, n_cols = self.input_shape
    #     n_directions = np.sqrt(n_channels)
    #     return (batch_size * n_directions * n_rows * n_cols, n_directions)