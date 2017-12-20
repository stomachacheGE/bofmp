

import numpy as np

class Grid_HMM(object):

    vel_to_idx = {-1: 0, 0: 1, 1: 2}
    idx_to_vel = {0: -1, 1: 0, 2: 1}

    def __init__(self, size, extent=3):
        # transition_counts has dimension of
        # [width, height, num_x_velocities, num_y_velocities, num_x_velocities, num_y_velocities]
        # dimension (2, 3) is last movement, (4, 5) is next movement
        self.size = size
        self.transition_counts = np.zeros(np.hstack([size, [extent]*4]))


    def add_transition(self, from_, current, to, add_symetric=True):

        last_mv_idx = self._pos_to_vel_idx(from_, current)
        next_mv_idx = self._pos_to_vel_idx(current, to)
        pos = current[0], current[1]
        idx = pos + last_mv_idx + next_mv_idx
        self.transition_counts[idx] += 1
        if add_symetric:
            # add symetric update
            self.add_transition(to, current, from_, add_symetric=False)

    def _pos_to_vel_idx(self, pos_1, pos_2):
        """ Velocity index of movement from pos_1 to pos_2. """
        vel_x = pos_2[0] - pos_1[0]
        vel_y = pos_2[1] - pos_1[1]
        return self.vel_to_idx[vel_x], self.vel_to_idx[vel_y]

    def get_transition_probs(self, conditional, diagonal):
        if conditional:
            axis = (4, 5)
        else:
            axis = (2, 3, 4, 5)

        with np.errstate(divide='ignore', invalid='ignore'):
            probs = self.transition_counts / self.transition_counts.sum(
                axis=axis, keepdims=True)
            probs[~np.isfinite(probs)] = 0

        if diagonal:
            velocities = [[0, 1], [1, 0], [0, -1], [-1, 0],
                          [1, 1], [1, -1], [-1, 1], [-1, -1]]
        else:
            velocities = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        vel_idxs = map(lambda vel: Grid_HMM.two_d_vel_to_idx(vel), velocities)
        # keep motion for empty entries
        for idx_xy in np.ndindex(tuple(self.size)):
            for idx_vel_last in vel_idxs:
                if np.all(probs[idx_xy + idx_vel_last] == 0):
                    probs[idx_xy + idx_vel_last + idx_vel_last] = 1

        return probs

    def get_transition_counts(self):
        return self.transition_counts

    @staticmethod
    def two_d_vel_to_idx(vel):
        return Grid_HMM.vel_to_idx[vel[0]], Grid_HMM.vel_to_idx[vel[1]]

    @staticmethod
    def two_d_idx_to_vel(idx):
        return Grid_HMM.idx_to_vel[idx[0]], Grid_HMM.idx_to_vel[idx[1]]

    @staticmethod
    def get_backward_vel(v1, v2):
        backward_v1 = [-v2[0], -v2[1]]
        backward_v2 = [-v1[0], -v1[1]]
        return backward_v1, backward_v2

    @staticmethod
    def get_unique_vel_idxs(velocities):
        vs = []
        backward_vs = []
        idxs = []
        backward_idxs = []
        for v1 in velocities:
            for v2 in velocities:
                idx1 = Grid_HMM.two_d_vel_to_idx(v1)
                idx2 = Grid_HMM.two_d_vel_to_idx(v2)
                if v1 == [-v2[0], -v2[1]]:
                    continue
                backward_v1, backward_v2 = Grid_HMM.get_backward_vel(v1, v2)
                b_idx1 = Grid_HMM.two_d_vel_to_idx(backward_v1)
                b_idx2 = Grid_HMM.two_d_vel_to_idx(backward_v2)
                v1_, v2_, backward_v1_, backward_v2_ = map(lambda foo: tuple(foo), [v1, v2, backward_v1, backward_v2])
                idx1_, idx2_, b_idx1_, b_idx2_ = map(lambda foo: tuple(foo), [idx1, idx2, b_idx1, b_idx2])
                if (v1_ + v2_) not in vs and (backward_v1_ + backward_v2_) not in vs:
                    vs.append((v1_ + v2_))
                    backward_vs.append((backward_v1_ + backward_v2_))
                if (idx1_ + idx2_) not in idxs and (b_idx1_ + b_idx2_) not in idxs:
                    idxs.append((idx1_ + idx2_))
                    backward_idxs.append((b_idx1_ + b_idx2_))
        return vs, idxs, backward_vs, backward_idxs


