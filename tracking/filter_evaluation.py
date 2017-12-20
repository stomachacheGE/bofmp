
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imp
import os
import time
import signal
from datetime import datetime

from utils import pickle_load, pickle_save, ensure_dir
from utils.occ_map_utils import load_map, display_occ_map, plot_grid_map_hmm, show_traj
from utils.occ_map_utils import show_map
from data_generator.ped_sim import sample_trajectories
from data_loader import get_map_crop
from bofum import conditionalBOFUM, naiveBOFUM
from test import get_model
from data_generator.ped_sim import sample_trajectories
from metrics import cross_entropy
from animation import TrackingAnimRealdata


_traj_options = {'speed': 1,
                 'constant_speed': False,
                 'diagonal': False
                 }


class BofumEvaluation(object):

    def __init__(self, num_steps, bofum_model, bofum_options, metrics, simulated_data=True,
                       cnn_model='', cache_folder='',
                       cnn_outputs_path=None):

        print("Evaluation starts at %s" % str(datetime.now()))

        self.num_steps = num_steps
        self.bofum = bofum_model
        self.bofum_options = bofum_options
        self.bofum_options['simulated_data'] = simulated_data
        self.bofum_options['model_name'] = cnn_model
        self.metrics = metrics
        self.cnn_model = cnn_model
        self.simulated_data = simulated_data
        self.name = bofum_options.get('name', None)
        self.cache_folder = cache_folder
        self.folder_path = self._get_folder_path()
        self.cnn_outputs_path = self.folder_path + '/cnn_output.npy'
        if cnn_outputs_path is not None:
            self.cnn_outputs_path = cnn_outputs_path
        self.maps = self._get_maps()
        self.models = self._get_models()
        self.num_models = self.maps.shape[0]
        self.results = {}
        for metric in self.metrics:
            self.results[metric] = np.zeros((self.num_models , self.num_steps), dtype=float)


    def _get_folder_path(self):
        """ Generate path for result folder."""
        pwd = os.path.dirname(os.path.realpath(__file__))
        folder = '/results/' if self.cache_folder == '' else '/results/{}/'.format(self.cache_folder)
        folder = pwd + folder
        ensure_dir(folder)
        return folder

    def _get_models(self):
        """ Create BOFUM models in which tracking can be done. """

        print("generating models...")
        t1 = time.time()
        options = self.bofum_options

        def create_model(map_, condi_probs=None):
            if condi_probs is not None:
                return self.bofum(map_, nn_probs=condi_probs, **options)
            else:
                return self.bofum(map_, **options)

        if self.bofum.__name__ == 'naiveBOFUM':
            options.pop("model_name")
            models = map(create_model, self.maps)
        elif self.bofum.__name__ == 'conditionalBOFUM':

            if os.path.isfile(self.cnn_outputs_path):
                cnn_outputs = np.load(self.cnn_outputs_path)
            else:
                cnn_outputs = self._get_cnn_output(self.maps, self.cnn_model)
                np.save(self.cnn_outputs_path, cnn_outputs)
            models = map(create_model, self.maps, cnn_outputs)
        t2 = time.time()
        print("model generation takes {:.3f} seconds".format(t2-t1))

        return models

    def _get_cnn_output(self, inputs, model_name):
        """ Get conditional probabilities for input maps from the trained CNN model. """

        outputs_path = self.cnn_outputs_path
        # if CNN output is there, load it
        if os.path.isfile(outputs_path):
            return np.load(outputs_path)
        # otherwise get it from CNN model
        model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))\
                     + '/trained_models/' + model_name
        cnn_model = get_model(model_path)
        # add color channel axis for network input
        inputs = np.expand_dims(inputs, axis=1)
        outputs = cnn_model(inputs)
        np.save(outputs_path, outputs)
        return outputs

    def initialize_model(self, model_idx):
        """This method has to be implemented by subclasses."""
        pass

    def _get_maps(self):
        """ Get maps from result folder or from test data set. """
        maps_path = self.folder_path + '/maps.npy'
        if os.path.isfile(maps_path):
            print("loading maps")
            maps = np.load(maps_path)
        else:
            maps = self._retrieve_maps()
            np.save(maps_path, maps)
        return maps

    def evaluate(self):
        """ Evaluate tracking performance. """



        print("tracking starts for {}...".format(self.name))
        t1 = time.time()

        map(lambda model_idx: self.initialize_model(model_idx), np.arange(self.num_models))

        # To see how tracking performs on each model
        # for _ in range(len(self.models)):
        #     ani = TrackingAnimRealdata([self.models[_]], self.num_steps, self.models[_].scene,
        #                                plot_map=True,
        #                                plot_seen=True)
        #     plt.show()

        # reset model before tracking evaluation
        map(lambda model: model.reset(), self.models)


        # tracking over time
        for t in range(self.num_steps):
            if t % 3 == 0:
                print("tracking {}/{} time steps".format(t+1, self.num_steps))

            map(lambda model: model.tracking_step(), self.models)
            # caculate cross entropy over the map
            self.add_result(t)

        t2 = time.time()
        print("tracking for {} takes {:.5f} seconds".format(self.name, t2-t1))


    def add_result(self, time_step):
        """ Add evaluation result for a specific time step."""

        for m in range(self.num_models):
            if 'x_ent' in self.metrics:
                self.results['x_ent'][m, time_step] = self.models[m].calc_cross_entropy()

            if 'f1_score' in self.metrics:
                self.results['f1_score'][m, time_step] = self.models[m].calc_f1_score()

            if 'average_precision' in self.metrics:
                self.results['average_precision'][m, time_step] = self.models[m].calc_average_precision()


    def get_results(self, metric=None):
        """ Get evaluation results. """

        params = ['extent', 'noise_var', 'omega']
        flag = '_'.join(map(lambda k: '{}_{}'.format(k, self.bofum_options[k]), params))
        file_path = self.folder_path+'/{}_{}_result.npy'.format(flag, self.metrics)
        if os.path.isfile(file_path):
            print("loading tracking results...")
            self.results = pickle_load(file_path)
        else:
            # do evaluation
            self.evaluate()
            prob_idx = self.filter_out_models_with_numerical_issue()
            for k, v in self.results.items():
                temp = np.array([v[i] for i in np.arange(self.num_models) if i not in prob_idx])
                self.results[k] = temp
            pickle_save(file_path, self.results)

        if metric is None:
            return self.results
        else:
            return self.results[metric]

    def filter_out_models_with_numerical_issue(self):
        prob_idx = []
        for i, model in enumerate(self.models):
            kernels = np.sum(model.kernels, axis=(-1, -2))
            if kernels.ndim == 2:
                if np.sum(kernels == 0) > 0:
                    prob_idx.append(i)
            elif kernels.ndim == 4:
                for idx_ in np.ndindex(kernels.shape[0], kernels.shape[1]):
                    if np.sum(kernels[idx_] == 0) > 0:
                        prob_idx.append(i)
                        break
        print("numerical issue model idx:{}".format(prob_idx))
        return prob_idx


class BofumEvaluationSimulation(BofumEvaluation):
    def __init__(self, num_evals, num_steps, bofum_model, bofum_options, metric, config_path,
                 map_repetition=5, num_targets=1,
                 cnn_model=None,
                 name='', cache_folder='', traj_options=_traj_options,
                 cnn_outputs_path=None):

        print("----------------{}---------------".format(name))
        self.config_path = config_path
        self.num_evals = num_evals
        self.map_repetition = map_repetition
        self.traj_options = traj_options
        self.num_targets = num_targets
        simulated_data = True
        super(BofumEvaluationSimulation, self).__init__(num_steps, bofum_model, bofum_options, metric, simulated_data,
                                                        cnn_model, cache_folder
                                                        , cnn_outputs_path)
        self.distances, self.trajs = self._get_trajs()

    def initialize_model(self, model_idx):
        model = self.models[model_idx]
        distances = self.distances[model_idx]
        trajectories = self.trajs[model_idx]
        model.initialize(self.num_targets, self.num_steps,
                         distances=distances,
                         trajectories=trajectories)

    def _retrieve_maps(self):
        # retrieve map crops from test set
        maps = get_map_crop(self.config_path, self.num_evals // self.map_repetition, dataset='test')
        # repeat maps
        maps = np.repeat(maps, self.map_repetition, axis=0)
        return maps

    def _get_trajs(self):
        """Get trajectories either from result folder or generate them."""

        distances_file_name, traj_file_name = '/distances.npy', '/trajectories.pkl'
        distances_path, traj_path = self.folder_path+distances_file_name, self.folder_path+traj_file_name
        if os.path.isfile(distances_path) and os.path.isfile(traj_path):
            print("loading trajectories...")
            distances, trajs = np.load(distances_path), pickle_load(traj_path)
        else:
            distances, trajs = self._generate_trajs()
            np.save(distances_path, distances)
            pickle_save(traj_path, trajs)
        return distances, trajs

    def _generate_trajs(self):
        """ Generate ground truth tracking trajectories in maps."""
        print("sampling trajectories...")
        distances_shape = (self.num_models, self.num_targets, self.num_steps)
        distances = np.zeros(distances_shape, dtype=int)
        trajs = []

        for i in range(self.num_models):
            if i % 5 == 0:
                print('Sampling trajectories on {}/{} models...'.format(i+1, self.num_models))
            # generate trajectories by initializing BOFUM model
            distances_, trajs_ = self.models[i].initialize(self.num_targets, self.num_steps, **self.traj_options)
            distances[i] = distances_
            trajs.append(trajs_)

        return distances, trajs

class BofumEvaluationRealdata(BofumEvaluation):

    def __init__(self, scenes, num_steps, bofum_model, bofum_options, metric,
                 cnn_model=None, cache_folder='', cnn_outputs_path=None, simulated_scenes=False):
        # print("----------------{}---------------".format(name))
        self.scenes = scenes
        simulated_data = False
        self.simulated_scenes = simulated_scenes
        super(BofumEvaluationRealdata, self).__init__(num_steps, bofum_model, bofum_options, metric, simulated_data,
                                                      cnn_model, cache_folder, cnn_outputs_path)

    def initialize_model(self, model_idx):
        model = self.models[model_idx]
        scene = self.scenes[model_idx]
        # simulated scenes don't need to preprocess
        model.initialize(scene, preprocessing=not self.simulated_scenes)

    def _retrieve_maps(self):
        # retrieve map from scene
        maps = np.array(map(lambda scene: scene.static_map, self.scenes))
        print("retrieved maps shape: {}".format(maps.shape))
        return maps


# class BofumEvaluationBaseline(BofumEvaluation):
#     """
#     Evaluate a BOFUM model which predicts uniform occupancy at every time step.
#     Uniform occupancy means every empty cell has a probability P{occupied} = 1 / num_emtpy_cell.
#     """
#     def evaluate(self):
#         """ Evaluate tracking performance. """
#         print("tracking starts for {}...".format(self.name))
#
#         def initialize_model(model_idx):
#             model = self.models[model_idx]
#             distances = self.distances[model_idx]
#             trajectories = self.trajs[model_idx]
#             model.initialize(self.num_targets, self.num_steps,
#                              distances=distances,
#                              trajectories=trajectories,
#                              )
#             # generate uniform occupancy assuming that
#             # total occupancy is uniformally
#             # distributed over all empty cells in the map
#             P_Ot = model._uniform_occupancy()
#             model.P_Ot = P_Ot
#
#         map(lambda model_idx: initialize_model(model_idx), np.arange(self.num_models))
#
#         # tracking over time
#         for t in range(self.num_steps):
#             if t % 3 == 0:
#                 print("tracking {}/{} time steps".format(t + 1, self.num_steps))
#
#             # tracking step
#             # simply do not track, and occupancy will stay as inititial state,
#             # which is uniform distribution
#
#             # caculate cross entropy over the map
#             self.add_result(t)

def show_results(evaluations, metric):
    """ Plot evaluation results with error bar."""
    fig, ax = plt.subplots()
    colors = cm.Dark2(np.linspace(0, 1, len(evaluations)))
    results = {}
    for i, evaluation in enumerate(evaluations):
        res = evaluation.get_results(metric)
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        ax.errorbar(np.arange(mean.shape[0]), mean, yerr=std, color=colors[i], label=evaluation.name, fmt='-o')
        results[evaluation.name] = res

    # store the results on disk
    pwd = os.path.dirname(os.path.realpath(__file__))
    folder = '/results/' if evaluations[0].cache_folder == '' else \
        '/results/{}/'.format(evaluations[0].cache_folder)
    folder = pwd + folder
    pickle_save(folder+'/measurement_lost_{}_{}_results.pkl'.format(evaluations[0].models[0].measurement_lost, metric), results)

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')

    plt.show()

if __name__ == "__main__":

    config_path = '/local/home/ful7rng/projects/transition/config.py'
    cnn_model_name = '23_ALL_MAPS_EIGHT_DIRECTIONS'

    bofum_options = { 'omega': 0.05,
                      'extent': 5,
                      'noise_var': 0.6
                      }

    traj_options = {'constant_speed': False,
                     'diagonal': True
                     }

    for num_evals_ in [20]:
        for num_targets_ in [1]:
            for measurement_lost_ in [None]:

                num_evals = num_evals_
                map_repetition = 5
                num_targets = num_targets_
                num_steps = 15
                measurement_lost = measurement_lost_

                result_folder_name = '43_{}_inputs_{}_repetition_{}_targets_{}_steps_eight_directions_no_keep_occupancy'.format(
                    num_evals, map_repetition, num_targets, num_steps)

                evaluations = []


                # naiveBOFUM
                naiveBofum_options = bofum_options
                naiveBofum_evaluation = BofumEvaluationSimulation(num_evals, num_steps,
                                                        naiveBOFUM, naiveBofum_options, config_path,
                                                        map_repetition, num_targets,
                                                        name = 'naiveBOFUM',
                                                        cache_folder=result_folder_name,
                                                        traj_options=traj_options)
                evaluations.append(naiveBofum_evaluation)

                # conditionalBOFUM
                # conditionalBofum_options = bofum_options.copy()
                # conditionalBofum_evaluation = BofumEvaluationSimulation(num_evals, num_steps,
                #                                               conditionalBOFUM, conditionalBofum_options, config_path,
                #                                               map_repetition, num_targets,
                #                                               measurement_lost=measurement_lost,
                #                                               name='conditionalBOFUM',
                #                                               cnn_model=cnn_model_name,
                #                                               cache_folder=result_folder_name,
                #                                               traj_options=traj_options)
                # evaluations.append(conditionalBofum_evaluation)

                # conditionalBOFUM with acceleration interpretation
                conditionalBofum_options_1 = bofum_options.copy()
                conditionalBofum_options_1['acceleration_interpretation'] = True
                conditionalBofum_evaluation_acc = BofumEvaluationSimulation(num_evals, num_steps,
                                                                  conditionalBOFUM, conditionalBofum_options_1,
                                                                  config_path,
                                                                  map_repetition, num_targets,
                                                                  name='conditionalBOFUM acc',
                                                                  cnn_model=cnn_model_name,
                                                                  cache_folder=result_folder_name,
                                                                  traj_options=traj_options)
                evaluations.append(conditionalBofum_evaluation_acc)

                # conditionalBOFUM with acceleration interpretation and condi_probs blurred
                conditionalBofum_options_2 = bofum_options.copy()
                conditionalBofum_options_2['acceleration_interpretation'] = True
                conditionalBofum_options_2['blur_spatially'] = True
                conditionalBofum_evaluation_acc_2 = BofumEvaluationSimulation(num_evals, num_steps,
                                                                  conditionalBOFUM, conditionalBofum_options_2,
                                                                  config_path,
                                                                  map_repetition, num_targets,
                                                                  name='conditionalBOFUM acc condi prob blurred',
                                                                  cnn_model=cnn_model_name,
                                                                  cache_folder=result_folder_name,
                                                                  traj_options=traj_options)
                evaluations.append(conditionalBofum_evaluation_acc_2)


                # uniformBOFUM as a baseline
                # Bofum_evaluation_baseline = BofumEvaluationBaseline(config_path, cnn_model_name, naiveBOFUM,
                #                                         num_evals, map_repetition, num_targets, num_steps,
                #                                         measurement_lost=measurement_lost,
                #                                         name = 'uniformBOFUM',
                #                                         cache_folder=result_folder_name,
                #                                         bofum_options=naiveBofum_options,
                #                                         traj_options=traj_options)
                # evaluations.append(Bofum_evaluation_baseline)


                show_results(evaluations)



