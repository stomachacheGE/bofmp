import numpy as np
import matplotlib.pyplot as plt


from occ_map_utils import show_map, plot_occ_map


def round_to_cell(trajectory, grid_origin, grid_res):
    # round each trajectory point to nearest grid cell center
    first_cell_centre = grid_origin + grid_res / 2
    trajectory = np.round((trajectory - first_cell_centre[None, :])/
                          grid_res).astype(int)
    # check for consecutive duplicates
    trajectory_ = [trajectory[0]]
    for ix, point in enumerate(trajectory[1:]):
        if not np.array_equal(point, trajectory[ix]):
            trajectory_.append(point)
    return np.array(trajectory_)


def plot_trajectories(trajectories, map_arr=None, map_res=1.0, map_origin=None):
    for trajectory in trajectories:
        plt.plot(trajectory[:, -2], trajectory[:, -1], zorder=11)
    if map_arr is not None:
        show_map(map_arr, resolution=map_res, origin=map_origin)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")


def ix2coords(trajectories, res, origin):
    if not isinstance(trajectories, list):
        if np.ndim(trajectories) == 1:
            trajectories = trajectories[None, ...]
        trajectories = [trajectories]
    out_trajs = []
    for trajectory in trajectories:
        # correct pixel offset
        trajectory[:, -2:] = trajectory[:, -2:] + 0.5
        # fix scale
        trajectory[:, -2:] *= res
        # shift to right origin
        trajectory[:, -2:] += origin[None, :]
        out_trajs.append(trajectory)
    return out_trajs


def coords2ix(trajectories, res, origin):
    if not isinstance(trajectories, list):
        if np.ndim(trajectories) == 1:
            trajectories = trajectories[None, ...]
        trajectories = [trajectories]
    out_trajs = []
    for trajectory in trajectories:
        # shift to right origin
        trajectory[:, -2:] -= origin[None, :]
        # fix scale
        trajectory[:, -2:] /= res
        # correct pixel offset
        trajectory -= 0.5
        out_trajs.append(trajectory)
    return out_trajs


def xy(trajectory):
    return np.array(trajectory)[..., 1:].astype(int)


def plot_occ_map_trajs(occ_map, trajectories, static_map=None, map_origin=None,
                       map_res=None):
    plot_occ_map(occ_map, map_origin=map_origin, map_res=map_res,
                 static_map=static_map)
    plot_trajectories(trajectories, map_origin=map_origin, map_res=map_res)