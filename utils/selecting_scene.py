

import os
from copy import deepcopy
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import Tkinter as Tk
import tkMessageBox
from Tkinter import N, W, E, S
import tkMessageBox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from propagation.animation import Plot
from propagation.bofum import BOFUMRealdata
from scene_utils import get_scenes
from utils import pickle_load, pickle_save

button_names = ['Multiple Objects', 'Objects with diff. motion', 'Turn', 'Straight', 'Straight and turn',
                'Turn and Straight', 'Through Doors', 'Exclude',
                'Strange Behaviour', 'Noisy', 'Missing Frames',]

colorbar_on = "occupancy_axes"

class SceneID(object):

    def __init__(self, file, idx):
        self.file = file
        self.idx = idx

def get_all_scenes_files(data_folder):
    file_list = []
    # get all .npy files
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".npy"):
                file_list.append(os.path.join(root, file))
    return file_list

def show_progress(cls):
    print("-------------------------------------")
    for k, v in cls.items():
        print("{}: {}".format(k, len(v)))

def init_app(data_folder):
    cls_path = os.path.join(data_folder, 'classification.pkl')
    if not os.path.isfile(cls_path):
        print("initialize classification result file")
        print("file is {}".format(cls_path))
        cls = dict()
        cls['processed'] = []
        for name in button_names:
            cls[name] = []
        pickle_save(cls_path, cls)
    else:
        print("load classification result file")
        cls = pickle_load(os.path.join(data_folder, 'classification.pkl'))
    all_files = get_all_scenes_files(data_folder)
    processed_files = cls['processed']
    remaining_files = [f for f in all_files if f not in processed_files]
    print("%d files remain to be processed" % len(remaining_files))
    show_progress(cls)
    return cls, cls_path, processed_files, remaining_files, all_files

def init_fig(scene):
    fig = plt.figure(figsize=(10, 5))
    ax_1, ax_2 = fig.subplots(1, 2)
    plot_1 = Plot(ax_1, scene.static_map, scene.res, plot_seen=False, title='Original Scene')
    plot_2 = Plot(ax_2, scene.static_map, scene.res, plot_seen=False, title='After preprocessing')
    fig_title_axes = fig.add_axes([.4, .92, .2, .05])
    fig_title_axes.set_axis_off()
    fig_title = fig.text(.49, .92, "", transform=fig_title_axes.transAxes, fontsize=12, color='r',
                                   ha='center')
    fig_title.set_text("Scene interval = {:.3f}s".format(scene.end - scene.start))
    return fig, [plot_1, plot_2], fig_title


class SceneAnimation(animation.TimedAnimation):

    def __init__(self, frames, fig, plots, scenes, t_gap, interval=500):
        self.frames = frames
        self.plots = plots
        self.scenes = scenes
        self.t_gap = t_gap
        super(SceneAnimation, self).__init__(fig, interval=interval, blit=True, repeat=True, repeat_delay=1000)

    def _draw_frame(self, frame):

        artist = []
        for scene_, plot in zip(self.scenes, self.plots):
            plot.set_axes_data("map_axes", scene_.static_map)
            plot.set_axes_data("occupancy_axes", scene_.hits[frame])
            plot.set_text("t={:.3f}s".format((frame+1) * self.t_gap))
            artist.append(plot.text)
            artist.append(plot.map_axes)
            artist.append(plot.occupancy_axes)

    def new_frame_seq(self):
        return iter(range(self.frames))

def init_animation(fig, row, column, root, colspan, rowspan):

    global scenes_to_plot, plots, scene, t_gap

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(column=column, row=row, columnspan=colspan, rowspan=rowspan)
    ani = SceneAnimation(scene.hits.shape[0], fig, plots, scenes_to_plot, t_gap, interval=150)

    return ani


def generate_buttons(texts, root):

    buttons = []
    for i, text in enumerate(texts):
        button = Tk.Button(root, text=text, command=lambda x=i: btn_pressed(x))
        button.grid(row=i+1, column=1, columnspan=2, sticky=W+E+S+N)
        buttons.append(button)

    return buttons

def update_animation():
    global scenes_to_plot, scene, t_gap

    ani.t_gap = t_gap
    ani.scenes = scenes_to_plot
    ani.frames = scene.hits.shape[0]
    ani.frame_seq = ani.new_frame_seq()
    ani.event_source.start()

def load_scenes(f_name):
    scenes = np.load(f_name)
    # sample rate 3
    for scene in scenes:
        num_frames = int(len(scene.hits)/3)
        x, y = scene.static_map.shape
        scene.hits = scene.hits[:num_frames * 3]
        scene.hits = np.sum(scene.hits.reshape(num_frames, 3, x, y), axis=1)
        scene.hits = np.where(scene.hits >= 1, 1, 0)
        # print(scene.hits.shape)
        # scene.hits = scene.hits[::3]
        # scene.seens = scene.seens[::3]
        #scene.seens = np.sum(scene.seens.reshape(3, -1), axis=0)


    return scenes

def next_clicked():

    global ani, flags, scene, scenes, all_files, num_processed, file_idx, scene_idx, \
        temp_cls, file_label, buttons, cls, t_gap, scenes_to_plot

    ani.event_source.stop()

    tags = []
    for i in range(len(flags)):
        if flags[i]:
            tags.append(button_names[i])
    print(' ; '.join(tags))

    scene_id = SceneID(remaining_files[file_idx-num_processed], scene_idx)

    for idx in range(len(flags)):
        if flags[idx]:
            temp_cls[button_names[idx]].append(scene_id)

    flags = np.zeros(len(buttons), dtype=bool)

    for button in buttons:
        button.configure(bg='#dcdad5')

    while True:

        if scene_idx == len(scenes) - 1:
            temp_cls['processed'] = [remaining_files[file_idx-num_processed]]
            add_temporary_to_cls()
            reset_temporary_results()

            if file_idx == len(all_files) - 1:
                tkMessageBox.showinfo("Congratulations", "You have finished all the files!")
                break
            else:
                file_idx += 1
                scene_idx = 0
                scenes = load_scenes(remaining_files[file_idx-num_processed])
                show_progress(cls)
        else:
            scene_idx += 1

        scene = deepcopy(scenes[scene_idx])
        scene_interval = scene.end - scene.start

        if 4 <= scene_interval < 5.5:
            break
        else:
            print("Skip scene (file_idx: {}; scene_idx: {}) : {:3f} seconds".format(file_idx+1, scene_idx+1, scene_interval))

    print("-------------------------------------")
    print("Scene lasts for {:.3f} seconds".format(scene.end - scene.start))
    fig_title.set_text("Scene interval = {:.3f}s".format(scene.end-scene.start))
    t_gap = (scene.end - scene.start) / scene.hits.shape[0]

    scene_copy = deepcopy(scene)
    BOFUMRealdata.scene_preprocessing(scene_copy)

    scenes_to_plot = [scene, scene_copy]

    file_label.configure(text='file: {}'.format(remaining_files[file_idx-num_processed]))
    f_label.configure(text="File: {}/{}".format(file_idx + 1, num_files))
    scene_label.configure(text='scene: {}/{}'.format(scene_idx + 1, len(scenes)))

    # update_animation
    update_animation()


def containSceneID(scene_ids, scene_id):
    for id_ in scene_ids:
        if id_.file == scene_id.file and id_.idx == scene_id.idx:
            # print("has scene_id {}/{}".format(scene_id.file, scene_id.idx))
            return True

def check_how_many_we_have(cls):
    scene_ids = np.array(cls['Straight and turn'] + cls['Straight'] + cls['Turn'] + cls['Turn and Straight'])
    print("Found %d scene_ids for 4 motion classes" % len(scene_ids))
    scene_ids_left = []
    for idx, id_ in enumerate(scene_ids):
        if not containSceneID(cls['Objects with diff. motion'], id_):
            if not containSceneID(scene_ids_left, id_):
                scene_ids_left.append(id_)
            else:
                # print("found duplicated scene_id {}/{}".format(id_.file, id_.idx))
                pass
    #print("Remove %d scene_ids since they have objects with different motion or they are duplicated" % ()
    return len(scene_ids_left)


def add_temporary_to_cls():
    global temp_cls, cls, cls_path
    for k, v in temp_cls.items():
        #print('{}: {}'.format(k, v))
        cls[k] += v
    num = check_how_many_we_have(cls)
    print("Found %d proper scenes: single object or multiple objects with same motion trend " % num)
    if num >= 500:
        tkMessageBox.showinfo("Congratulations", "You have more than 500 proper scenes already!")
    pickle_save(cls_path, cls)

def reset_temporary_results():
    global temp_cls
    temp_cls = init_temp_cls()

def init_temp_cls():
    temp_cls = dict()
    temp_cls['processed'] = []
    for name in button_names:
        temp_cls[name] = []
    return temp_cls

if __name__ == '__main__':

    temp_cls = init_temp_cls()

    data_folder = '/local/data/scenes/80_new'
    # data_folder = '/home/ful7rng/projects/transition/data/test_scenes'
    cls, cls_path, processed_files, remaining_files, all_files = init_app(data_folder)
    num_processed = len(processed_files)

    root = Tk.Tk()
    if num_processed == len(all_files):
        tkMessageBox.showinfo("Congratulations", "You have finished all the files!")
    else:
        num_files = len(all_files)
        file_idx, scene_idx = num_processed, 0
        scenes = load_scenes(remaining_files[file_idx-num_processed])
        scene = deepcopy(scenes[scene_idx])
        t_gap = (scene.end - scene.start) / scene.hits.shape[0]

        fig, plots, fig_title = init_fig(scene)
        scene_copy = deepcopy(scene)
        BOFUMRealdata.scene_preprocessing(scene_copy)
        scenes_to_plot = [scene, scene_copy]

        f_label_text= "File: {}/{}".format(file_idx+1, num_files)
        f_label = Tk.Label(root, text=f_label_text, font=('Times', '10', 'bold italic'))
        f_label.grid(column=2, row=0)
        file_label = Tk.Label(root, text=remaining_files[file_idx-num_processed])
        file_label.grid(column=0, row=0)
        scene_label_text= "scene: {}/{}".format(scene_idx+1, len(scenes))
        scene_label = Tk.Label(root, text=scene_label_text, font=('Times', '10', 'bold italic'))
        scene_label.grid(column=1, row=0)
        # scene_label.pack(side='right')
        ani = init_animation(fig, 1, 0, root, 1, len(button_names) - 1)
        buttons = generate_buttons(button_names, root)
        flags = np.zeros(len(buttons), dtype=bool)
        next_button = Tk.Button(root, text='Next', command=next_clicked, height=3)
        next_button.grid(row=len(buttons), column=0, columnspan=1, sticky=S + E + W + N)


        def btn_pressed(i):
            button = buttons[i]
            flags[i] ^= True
            if button.cget('bg') == 'red':
                button.configure(bg='#dcdad5')
            else:
                button.configure(bg='red')

    Tk.mainloop()