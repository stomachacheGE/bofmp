
import numpy as np
from lasagne.layers import get_all_param_values, set_all_param_values, \
    get_all_layers, get_output_shape, InputLayer, Conv2DLayer, Pool2DLayer, \
    TransposedConv2DLayer, ConcatLayer

class Network():
    def __init__(self):
        # add architecture here
        pass

    def save(self, path):
        """ Save the weights """
        np.savez(path, *get_all_param_values(self.output_layer))

    def restore(self, path):
        """ Load the weights """

        with np.load(path) as f:
            saved_params_values = [f['arr_%d' % i] for i in range(len(f.files))]
        set_all_param_values(self.output_layer, saved_params_values)

    def summary(self, light=False):
        """ Print a summary of the network architecture """

        layer_list = get_all_layers(self.output_layer)

        def filter_function(layer):
            """ We only display the layers in the list below"""
            return np.any([isinstance(layer, layer_type) for layer_type in
                           [InputLayer, Conv2DLayer, Pool2DLayer, TransposedConv2DLayer, ConcatLayer]])

        layer_list = [layer for layer in layer_list if filter_function(layer)]
        output_shape_list = map(get_output_shape, layer_list)
        layer_name_function = lambda s: str(s).split('.')[3].split('Layer')[0]

        if not light:
            print('-' * 75)
            print('Warning : all the layers are not displayed \n')
            print('    {:<15} {:<20} {:<20}'.format('Layer', 'Output shape', 'W shape'))

            for i, (layer, output_shape) in enumerate(zip(layer_list, output_shape_list)):
                if hasattr(layer, 'W'):
                    input_shape = layer.W.get_value().shape
                else:
                    input_shape = ''

                print('{:<3} {:<15} {:<20} {:<20}'.format(i + 1,
                                                          layer_name_function(layer), str(output_shape), str(input_shape)))
                if isinstance(layer, Pool2DLayer) | isinstance(layer, TransposedConv2DLayer):
                    print('')

        print('\nNumber of Convolutional layers : {}'.format(
            len([x for x in layer_list if isinstance(x, Conv2DLayer) |
                 isinstance(x, TransposedConv2DLayer)])))
        print('Number of parameters : {}'.format(
            np.sum([np.size(pars) for pars in get_all_param_values(
                self.output_layer)])))
        print('-' * 75)