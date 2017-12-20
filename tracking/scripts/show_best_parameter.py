

from utils import pickle_load
from matplotlib import cm
import matplotlib.pyplot as plt
import collections


def show_results(res_paths):
    results = {}
    for path in res_paths:
        result = pickle_load(path)
        for k, v in result.items():
            if k not in results.keys():
                results[k] = result[k]
    results = collections.OrderedDict(sorted(results.items()))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = cm.Dark2(np.linspace(0, 1, len(results)))
    count = 0
    for k, res in results.items():
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        # ax.errorbar(np.arange(mean.shape[0]), mean, yerr=std, color=colors[count], label=k, fmt='-o')
        plt.plot(np.arange(mean.shape[0]) + 1, mean, '-o', color=colors[count], label=k)
        count += 1
        print(np.array_str(mean[8:], precision=3))
        print("Average precision of %s for future prediction: %f" % (k, mean[8:].mean()))

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper right')

    ax.set_xlabel("time step")
    ax.set_ylabel("average precision")

    plt.axvline(x=8.5, color='r', linestyle='--')
    plt.text(3, 0.1, 'tracking', fontsize=18, color='grey')
    plt.text(11, 0.1, 'prediction', fontsize=18, color='grey')

    plt.show()

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
    #print(items[0])

    items = sorted(items, key=lambda item: item[metric + ' mean'])
    if metric == 'f1_score' or metric == 'average_precision':
        items = items[::-1]

    for i in range(k):
        print("------- {}th best ------- ".format(i+1))
        print_dict(items[i])
