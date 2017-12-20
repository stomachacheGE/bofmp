import numpy as np
import os
from utils import ensure_dir
from utils.scene_utils import get_scenes, get_simulated_scenes


if __name__ == '__main__':

    laser_frequency = 12
    sample_rate = 3
    min_time_interval = 4
    num_steps = int(laser_frequency * min_time_interval / sample_rate)

    scenes_folder = os.path.dirname(os.path.realpath(__file__)) + '/scenes/'
    print(scenes_folder)
    ensure_dir(scenes_folder)

    real_scene_file = '/home/ful7rng/projects/transition/data/500_scenes_from_80_new.npy'
    scene_file = os.path.join(scenes_folder, 'scenes.npy')

    num_scenes = 500

    scenes, return_flag = get_scenes('/home/ful7rng/projects/transition/data/',
                                     0, 1e6,
                                     max_scenes=num_scenes*1.5,
                                     file_name=real_scene_file,
                                     sample_rate=sample_rate,
                                     laser_fre=laser_frequency)

    maps = [scene.static_map for scene in scenes]

    scenes = get_simulated_scenes(maps, num_steps, num_scenes, diagonal=True)
    np.save(scene_file, scenes)