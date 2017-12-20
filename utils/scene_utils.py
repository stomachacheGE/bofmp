import os
from random import sample

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.occ_map_utils import display_occ_maps
from utils.trajectory_utils import plot_trajectories
from tracking.animation import Plot

class Scene(object):
    def __init__(self, static_map, resolution, origin, start, end, hits, seens):
        self.static_map = static_map
        self.res = resolution
        self.origin = origin
        self.start = start
        self.end = end
        self.hits = hits
        self.seens = seens


def display_scenes_trajs(scenes, trajs):
    for scene_ix, scene in enumerate(scenes):
        num_plots = 5
        num_time_steps = len(scene.hits)
        step_size = (scene.end - scene.start) / num_time_steps
        plot_steps = num_time_steps // num_plots
        sel_ixs = np.arange(0, num_time_steps, plot_steps)
        plot = display_occ_maps(
            scene.hits[::plot_steps, None, ...],
            static_map=scene.static_map,
            col_labels=['t = {:.2} s'.format(sel_ix * step_size) for sel_ix in
                        sel_ixs],
            map_origin=scene.origin, resolution=scene.res)

        plt.sca(plot.axes)
        xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()

        # current_trajs = [traj for traj in trajs if
        #                  traj[0, 0] <= scene.start and
        #                  traj[-1, 0] >= scene.end]
        current_trajs = [trajs[scene_ix]]
        print("found %d trajs" % len(current_trajs))
        plot_trajectories(current_trajs)


        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)

        plt.show()

class SceneAnimation(animation.TimedAnimation):

    def __init__(self, scene, interval=50):
        self.frames = scene.hits.shape[0]
        self.scene = scene
        self.t_gap = (scene.end - scene.start) / scene.hits.shape[0]
        self.init_fig()
        super(SceneAnimation, self).__init__(self.fig, interval=interval, blit=True, repeat=True, repeat_delay=200)

    def _draw_frame(self, frame):
        artist = []
        self.plot.set_axes_data("map_axes", self.scene.static_map)
        self.plot.set_axes_data("occupancy_axes", self.scene.hits[frame])
        self.plot.set_text("t={:.3f}s".format((frame+1) * self.t_gap))
        artist.append(self.plot.text)
        artist.append(self.plot.map_axes)
        artist.append(self.plot.occupancy_axes)

    def init_fig(self):
        self.fig = plt.figure(figsize=(6, 5))
        ax = self.fig.add_subplot(111)
        title = "Scene interval = {:.3f}s".format(self.scene.end - self.scene.start)
        self.plot = Plot(ax, self.scene.static_map, self.scene.res, plot_seen=False, title=title)

    def new_frame_seq(self):
        return iter(range(self.frames))

def animate_scenes(scenes, need_return=False, save_path=None):

    for scene_ix, scene in enumerate(scenes):

        ani = SceneAnimation(scene)

        if save_path is not None:
            ani.save(save_path, writer='ffmpeg')

        if not need_return:
            plt.show()

    return ani

def get_scenes(data_folder='/local/data/scenes/100_11', min_time_interval=2,
               max_time_interval=1e6, random_file=False,
               max_scenes=None, laser_fre=12, file_name=None, sample_rate=3,
               simulated_scenes=False):

    return_flag = 'all_data'

    if file_name is not None:
        print("load file: %s" % file_name)
        if simulated_scenes:
            data = np.load(file_name)
            if max_scenes is not None and len(data) > max_scenes:
                data = np.array(sample(data, max_scenes))
            print("Take %d scenes for evaluation" % len(data))
            return data, return_flag
        else:
            data = [np.load(file_name)]
    else:
        file_list = []
        names = []
        # get all .npy files
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".npy"):
                    file_list.append(os.path.join(root, file))
                    names.append(file.split('.')[-2])

        print("Found %d files in total" % len(file_list))

        if random_file:
            random_idx = np.random.randint(0, len(file_list))
            random_file = file_list[random_idx]
            print("load file: %s" % random_file)
            data = [np.load(random_file)]
            return_flag = names[random_idx]
        else:
            data = map(lambda file_: np.load(file_), file_list)

    num_scenes = map(lambda scenes: scenes.shape[0], data)
    intervals = map(lambda scenes: map(lambda scene: scene.end - scene.start, scenes), data)
    total_scenes = sum(num_scenes)
    print("Found %d scenes in total" % total_scenes)

    # get scene whose time interval longer than min_time_interval
    eval_data = []
    idxs_ = []
    problematic = []
    min_steps = int(min_time_interval * laser_fre)
    max_steps = int(max_time_interval * laser_fre)
    print(min_steps)
    print(max_steps)
    for file_idx, interval_list in enumerate(intervals):
        for scene_idx, interval in enumerate(interval_list):
            if min_time_interval <= interval <= max_time_interval:
                # numbers of hits do not match time interval
                if min_steps <= data[file_idx][scene_idx].hits.shape[0] <= max_steps:
                    idxs_.append((file_idx, scene_idx))
                    eval_data.append(data[file_idx][scene_idx])
                else:
                    problematic.append((file_idx, scene_idx))

    eval_data = np.array(eval_data)

    for scene in eval_data:
        new_shape = scene.hits.shape[0]
        if new_shape == 1:
            print("{}".format(new_shape))

    num_scenes_needed = len(idxs_)
    print("Found %f scenes (= %.2f%%) whose interval is in range (%.2f, %.2f)" % (
        num_scenes_needed, float(num_scenes_needed) / total_scenes * 100, min_time_interval, max_time_interval))
    num_problematic_scenes = len(problematic)
    print("Found %d scenes (= %.3f%%) whose interval does not match number of hits." % (
        num_problematic_scenes, float(num_problematic_scenes) / total_scenes * 100))

    if max_scenes is not None and num_scenes_needed > max_scenes:
        eval_data = np.array(sample(eval_data, max_scenes))
        print("Take %d scenes for evaluation" % max_scenes)

    if sample_rate is not None:
        for scene in eval_data:
            # scene.hits = scene.hits[::sample_rate]
            # scene.seens = scene.seens[::sample_rate]
            num_frames = int(len(scene.hits) / sample_rate)
            x, y = scene.static_map.shape

            def accumulate_frames(frames):
                frames = frames[:num_frames * sample_rate]
                frames = np.sum(frames.reshape(num_frames, sample_rate, x, y), axis=1)
                frames = np.where(frames >= 1, 1, 0)
                return frames

            scene.hits = accumulate_frames(scene.hits)
            scene.seens = accumulate_frames(scene.seens)

    return_flag += "_longer_than_{}_max_scenes_{}".format(min_time_interval, max_scenes)
    return eval_data, return_flag

def get_simulated_scenes(maps, num_steps, num_scenes, diagonal):

    from propagation.bofum import naiveBOFUM

    def create_model(map_):
        # noise_var determines how likely an object accelerates
        return naiveBOFUM(map_, noise_var=0.4, simulated_data=True)

    models = map(create_model, maps)

    def get_scene(model):
        num_targets = np.random.choice([1, 2], p=[0.7, 0.3])
        model.initialize(num_targets, num_steps, diagonal=diagonal)
        #animate_scenes([model.trajs_to_scene()])
        return model.trajs_to_scene()

    scenes = []
    count = 0
    for idx, model in enumerate(models):
        if count == num_scenes:
            break

        print("generating scenes {}/{}".format(idx + 1, num_scenes))
        try:
            scene = get_scene(model)
            scenes.append(scene)
            count += 1
        except Exception as e:
            print("Unabled to generate scene due to: {}".format(e))
            print("Skip this map.")
            plt.imshow(model.map)
            plt.show()

    return scenes

def caculate_traj_from_scene(scene):
    # assume only one object in scene
    num_steps = scene.hits.shape[0]
    traj = np.zeros((num_steps, 2))
    for i in range(num_steps):
        hit = scene.hits[i]
        locs = np.array(np.where(hit))
        xs, ys = locs[0], locs[1]
        min_x, max_x = xs.min(), xs.max()+1
        min_y, max_y = ys.min(), ys.max()+1
        traj[i] = [(float(min_x)+max_x)/2*scene.res, (float(min_y)+max_y)/2*scene.res]
    return traj


if __name__ == "__main__":

    # file_list = []
    # # get all .npy files
    # for root, dirs, files in os.walk("/local/data/scenes"):
    #     for file in files:
    #         if file.endswith(".npy"):
    #             file_list.append(os.path.join(root, file))
    #
    # f = np.random.choice(file_list)
    # print(f)
    #
    # scenes = get_npy(get_file(f, 'scenes'), None)
    # trajs = get_npy(get_file(f, 'trajectories/real'), None)
    # print(scenes.shape)
    # print("Found %d scenes." % scenes.shape[0])
    # print("Found %d trajectories." % trajs.shape[0])

    scenes, _ = get_scenes(data_folder="/local/data/scenes/100_11", min_time_interval=3, max_time_interval=4)
    print(scenes.shape)
    animate_scenes(scenes)