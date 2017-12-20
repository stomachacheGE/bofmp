
import numpy as np
from copy import copy

from tracking.filters import naiveBOFUM, conditionalBOFUM
from tracking.visualize import VisualizeSimulation
from tracking.scripts.visualize_tracking import map_portfolio


if __name__ == "__main__":

    ######################## NOTE #########################################
    # To reproduce the plot in the thesis, you need to uncomment
    # "modify acceleration" part of conditionalBOFUM.blur_on_accleration
    #######################################################################

    cnn_model = '04_DIAGONAL_TRUE_CONDITIONAL_TRUE'
    nn_probs_path = "/home/ful7rng/Desktop/thesis/writting_materials/91_T_section_probs_cost_1_2_cost_2_1.npy"
    num_steps = 30

    map_, width, height, x, y = map_portfolio('big_T_section')

    filters = []

    kwargs = {
        'noise_var': .5,
        'omega': 0.2,
        'extent': 5}

    ###########################################
    ############### BOFUM #####################
    ###########################################
    bofum_kwargs = copy(kwargs)
    bofum_kwargs['name'] = 'BOFUM'
    bofum = naiveBOFUM(map_, simulated_data=True, **bofum_kwargs)
    filters.append(bofum)

    ###########################################
    ############### BOFMP #####################
    ###########################################
    bofmp_kwargs = copy(kwargs)
    bofmp_kwargs['name'] = 'BOFMP'
    nn_probs = np.load(nn_probs_path)
    bofmp = conditionalBOFUM(map_, cnn_model,
                             simulated_data=True,
                             force_predict=False,
                             nn_probs = nn_probs,
                              **bofmp_kwargs)
    filters.append(bofmp)

    ##### propose initial location of an object for T-section map
    width = 91
    anchor = np.array([width * 5 / 6, width / 2])
    dists = [np.array([x, y]) for x in (0, 1) for y in (-1, 0, 1)]
    init_locs = map(lambda dist: anchor + dist, dists)

    visualization = VisualizeSimulation(filters, num_steps, num_targets=1,
                                        dynamically=False,
                                        show_map=True,
                                        num_col=3,
                                        show_colorbar=False,
                                        init_locs=init_locs,
                                        init_moves=['LEFT_LEFT_LEFT'] * len(dists),
                                        show_at = [0, 15, 30])

    visualization.start()

