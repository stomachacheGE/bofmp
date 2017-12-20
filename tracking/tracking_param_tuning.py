import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imp
import os
from multiprocessing import Queue, Pool, Manager, Value, Lock
import resource

from utils import ensure_dir
from utils.scene_utils import get_scenes
from filters import conditionalBOFUM, naiveBOFUM
from filter_evaluation import BofumEvaluationRealdata
import argparse

best = np.inf


def mem(id):
    print('Memory usage on process %d      : % 2.2f MB' % (id, round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0,1))
    )


def random_param(keep_motion, blur_spatially):
    params = {}
    params['extent'] = np.random.choice([3, 5, 7])
    params['noise_var'] = np.random.uniform(0.1, 0.8)
    params['omega'] = np.random.uniform(0.01, 0.2)
    # params['extent'] = 5
    # params['noise_var'] = 0.165
    # params['omega'] = 0.064
    if keep_motion:
        params['window_size'] = np.random.choice([2, 4, 6])
        params['keep_motion_factor'] = np.random.uniform(0.3, 0.8)
        params['initial_motion_factor'] = np.random.uniform(0.3, 0.8)
    if blur_spatially:
        params['blur_extent'] = np.random.choice([3, 5, 7, 9])
        params['blur_var'] = np.random.uniform(0.5, 2)
    return params


def eval_with_params(param_id, tries, scenes, num_steps, measurement_lost, bofum, params, simulated_scenes):

    metrics = ['x_ent', 'f1_score', 'average_precision']

    bofum_options = params

    cnn_model = None
    num_scenes = len(scenes)
    if 'cnn_model' in params:
        result_folder_name = '{}/simulated_scenes_{}_{}_num_scenes_{}_tries_{}/{}'.format(
                                                                      params['cnn_model'],
                                                                      simulated_scenes,
                                                                      params['name'],
                                                                      num_scenes, tries,
                                                                      param_id)
        cnn_model = params.pop('cnn_model')
    else:
        result_folder_name = '{}/simulated_scenes_{}_num_scenes_{}_tries_{}/{}'.format(params['name'],
                                                                                       simulated_scenes,
                                                                                       num_scenes, tries, param_id)
    summary_folder_name = '/'.join(result_folder_name.split('/')[:-1])
    summary_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/' + summary_folder_name
    ensure_dir(summary_folder)
    cnn_outputs_path = '{}/{}_output.npy'.format(summary_folder, cnn_model)

    #
    evaluation = BofumEvaluationRealdata(scenes, num_steps,
                                                  bofum, bofum_options, metrics,
                                                  cnn_model=cnn_model,
                                                  cache_folder=result_folder_name,
                                                  cnn_outputs_path=cnn_outputs_path,
                                                  simulated_scenes=simulated_scenes)

    results = evaluation.get_results()
    for k, v in results.items():
        # value length is 0 means there are numerical
        # issue with this set of parameters
        if len(v) == 0:
            flag = None
            if k == 'x_ent':
                flag = np.inf
            elif k == 'f1_score' or k == 'average_precision':
                flag = 0
            results[k] = np.array([flag])

    results_in_file = {}
    for metric_name, res in results.items():

        if len(res) > 1:
            # caculate nanmean over all scenens
            # for x_ent, nan exists when ground truth does not intersect with seen
            # for average precision, nan exist when seen is all zeros
            mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
            # get mean cross entropy five steps after measurement lost
            mean_steps = mean[measurement_lost:]
            result = mean_steps.mean()
        else:
            mean_steps = res
            result = res[0]
        results_in_file[metric_name] = mean_steps
        results_in_file[metric_name+' mean'] = result


    # save to file
    save_file = summary_folder + '/summary.csv'
    print(save_file)
    # params.pop('config')
    if not os.path.isfile(save_file):
        with open(save_file, 'w') as f_handle:
            titles = ['id'] + params.keys() + results_in_file.keys()
            f_handle.write(','.join(titles) + '\n')

    with open(save_file, 'a') as f_handle:
        for metric_name in metrics:
            results_in_file[metric_name] = np.array_str(results_in_file[metric_name], precision=3)
            results_in_file[metric_name+' mean'] = str(results_in_file[metric_name+' mean'])
        values = [str(param_id)] + map(str, params.values()) + results_in_file.values()
        f_handle.write(','.join(values) + '\n')

    return save_file

def show_best(filename, metric, k=1):

    def line_to_list(line):
        exclude_next_line = lambda x: x[:-1] if x.endswith('\n') else x
        entries = map(exclude_next_line, line.split(','))
        return entries

    items = []

    def print_dict(dic, attrs=None):

        if attrs is None:
            attrs = ['omega', 'noise_var', 'extent', metric, metric + ' mean']
            if 'keep_motion' in dic and dic['keep_motion']:
                attrs += ['window_size', 'initial_motion_factor', 'keep_motion_factor']
            if 'blur_spatially' in dic and dic['blur_spatially']:
                attrs += ['blur_extent', 'blur_var']

        for k, v in dic.items():
            if attrs is not None and k not in attrs:
                continue
            print("{}: {}".format(k, v))

    with open(filename, 'r') as f:
        line = f.readline()
        #print(line)
        attrs = line_to_list(line)

        for i, line in enumerate(f):
            #print(line)
            values = line_to_list(line)
            #print(values)
            dict_ = {k: v for (k, v) in zip(attrs, values)}
            items.append(dict_)
    print(items[0])

    items = sorted(items, key=lambda item: item[metric + ' mean'])
    if metric == 'f1_score' or metric == 'average_precision':
        items = items[::-1]

    for i in range(k):
        print("------- {}th best ------- ".format(i+1))
        print_dict(items[i])


def worker_main(queue, num_tries, scenes, num_steps, measurement_lost, bofum, simulated_scenes):


    if queue.empty():
        return ("Finished all the jobs")

    job_id, params = queue.get(True)
    print("-------------------------------")
    print("Job {} is working on process {} ".format(job_id, os.getpid()))
    for key, value in params.items():
        print("{}:{}".format(key, value))
    # print(params)
    summary_file = eval_with_params(job_id, num_tries, scenes, num_steps,
                                    measurement_lost, bofum, params, simulated_scenes)

    mem(os.getpid())

    return summary_file


def parameter_tuning(cnn_model, metric, model_options, measurement_lost, num_scenes, simulated_scenes,
                     simulated_scenes_diagonal=None,
                     bofum=conditionalBOFUM,
                     num_tries=100, min_time_interval=3, max_time_interval=1e6,
                     sample_rate=3, num_subprocess=15, scene_folder='/local/data/scenes/100_11',
                     scene_file=None):

    laser_frequency = 12
    num_steps = int(laser_frequency * min_time_interval / sample_rate)

    scenes_filename = "scenes_num_{}_min_t_{}_max_t_{}_sample_rate_{}_simulated_{}_simluated_diagonal_{}.npy".format(num_scenes,
                                                                   min_time_interval,
                                                                   max_time_interval,
                                                                   sample_rate,
                                                                   simulated_scenes,
                                                                   simulated_scenes_diagonal)
    print("Loading scenes from : %s " % scenes_filename)
    scenes_path = os.path.dirname(os.path.realpath(__file__)) + '/results/'+scenes_filename
    if os.path.isfile(scenes_path):
        scenes = np.load(scenes_path)
    else:
        scenes, return_flag = get_scenes(scene_folder, min_time_interval, max_time_interval,
                                         max_scenes=num_scenes,
                                         file_name=scene_file,
                                         sample_rate=sample_rate,
                                         laser_fre=laser_frequency,
                                         simulated_scenes=simulated_scenes)
        #scenes = np.load(scene_file)
        np.save(scenes_path, scenes)

    m = Manager()
    param_queue = m.Queue()

    for i in range(num_tries):
        params = random_param(keep_motion=model_options.get('keep_motion', False),
                              blur_spatially=model_options.get('blur_spatially', False))
        if bofum == conditionalBOFUM:
            params['cnn_model'] = cnn_model
        params['measurement_lost'] = measurement_lost
        for k, v in model_options.items():
            params[k] = v
        flag = '_'.join(map(lambda k, v: '{}_{}'.format(k, v),
                            model_options.keys(),
                            model_options.values()))
        params['name'] = '{}_{}'.format(bofum.__name__, flag)
        param_queue.put((i, params))

    args = (param_queue, num_tries, scenes, num_steps, measurement_lost, bofum, simulated_scenes)
    #
    # for i in range(num_tries):
    #     # evluate once for generating cnn ouputs
    #     summary_file = worker_main(*args)

    # evluate once for generating cnn ouputs
    summary_file = worker_main(*args)

    pool = Pool(num_subprocess, maxtasksperchild=2)
    workers = [pool.apply_async(worker_main, args) for i in range(num_tries)]
    for worker in workers:
        print(worker.get())

    show_best(summary_file, metric, k=5)


