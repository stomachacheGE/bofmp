
import theano.tensor as T
from lasagne.init import HeUniform
from lasagne.layers import InputLayer, ConcatLayer, Conv2DLayer
from lasagne.nonlinearities import linear

from architectures import Network
from layers import BN_ReLU_Conv, TransitionDown, TransitionUp, \
    TransitionalSoftmaxLayer, TransitionalNormalizeLayer, SoftmaxLayer


class FCDenseNet(Network):
    def __init__(self,
                 input_shape=(None, 3, None, None),
                 n_directions = 4,
                 n_filters_first_conv=48,
                 n_pool=4,
                 growth_rate=12,
                 n_layers_per_block=5,
                 dropout_p=0.2,
                 conditional=False):
        """
        This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
        The network consist of a downsampling path, where dense blocks and transition down are applied, followed
        by an upsampling path where transition up and dense blocks are applied.
        Skip connections are used between the downsampling path and the upsampling path
        Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.

        :param input_shape: shape of the input batch. Only the first dimension (n_channels) is needed
        :param n_directions: number of drictions considered in the transitional model
        :param n_filters_first_conv: number of filters for the first convolution applied
        :param n_pool: number of pooling layers = number of transition down = number of transition up
        :param growth_rate: number of new feature maps created by each layer in a dense block
        :param n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
        :param dropout_p: dropout rate applied after each convolution (0. for not using)
        """

        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
        else:
            raise ValueError

        # Theano variables
        self.input_var = T.tensor4('input_var', dtype='float32')  # input image
        self.target_var = T.tensor4('target_var', dtype='float32')  # target

        #####################
        # First Convolution #
        #####################

        inputs = InputLayer(input_shape, self.input_var)

        # We perform a first convolution. All the features maps will be stored in the tensor called stack (the Tiramisu)
        stack = Conv2DLayer(inputs, n_filters_first_conv, filter_size=1, pad='same',
                            W=HeUniform(gain='relu'),
                            nonlinearity=linear, flip_filters=False)
        # The number of feature maps in the stack is stored in the variable n_filters
        n_filters = n_filters_first_conv

        #####################
        # Downsampling path #
        #####################

        skip_connection_list = []

        for i in range(n_pool):
            # Dense Block
            for j in range(n_layers_per_block[i]):
                # Compute new feature maps
                l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
                # And stack it : the Tiramisu is growing
                stack = ConcatLayer([stack, l])
                n_filters += growth_rate
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            stack = TransitionDown(stack, n_filters, dropout_p)

        skip_connection_list = skip_connection_list[::-1]

        #####################
        #     Bottleneck    #
        #####################

        # We store now the output of the next dense block in a list. We will only upsample these new feature maps
        block_to_upsample = []

        # Dense Block
        for j in range(n_layers_per_block[n_pool]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = ConcatLayer([stack, l])


        #######################
        #   Upsampling path   #
        #######################

        for i in range(n_pool):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

            # Dense Block
            block_to_upsample = []
            for j in range(n_layers_per_block[n_pool + i + 1]):
                l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
                block_to_upsample.append(l)
                stack = ConcatLayer([stack, l])

        #####################
        #      Sigmoid      #
        #####################

        if conditional:
            self.output_layer = TransitionalSoftmaxLayer(stack, n_directions)
        else:
            self.output_layer = SoftmaxLayer(stack, n_directions)



    ################################################################################################################



if __name__ == '__main__':
    FCDenseNet(input_shape=(5, 1, 32, 32),
            n_pool=2,
            n_classes=2,
            n_layers_per_block=3,
            growth_rate=6).summary()
