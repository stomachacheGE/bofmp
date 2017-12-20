
from tracking.filter_evaluation import BofumEvaluationRealdata

if __name__ == "__main__":

    config_path = '/local/home/ful7rng/projects/transition/config.py'
    cnn_model_name = '04_DIAGONAL_TRUE_CONDITIONAL_TRUE'
    laser_frequency = 12
    min_time_interval = 4
    sample_rate = 3
    num_steps = int(laser_frequency * min_time_interval / sample_rate)

    simulated_scene = False
    simulated_scenes_diagonal = True

    if not simulated_scene:
        scene_file = "/home/ful7rng/projects/transition/data/244_scenes_for_test_from_80_new.npy"
        scenes, return_flag = get_scenes(min_time_interval=min_time_interval,
                                         file_name=scene_file,
                                         laser_fre=laser_frequency)
    else:
        if simulated_scenes_diagonal:
            scene_file = '/home/ful7rng/projects/transition/propagation/scenes/500_simulated_scenes_diagonal_for_testing_from_80_new.npy'
        else:
            scene_file = '/home/ful7rng/projects/transition/propagation/scenes/500_simulated_scenes_not_diagonal_for_testing_from_80_new.npy'
        scenes, return_flag = get_scenes(file_name=scene_file, simulated_scenes=simulated_scene)

    naive_bofum_options = {
                        'omega': 0.152265835237,
                        'extent': 5,
                        'noise_var': 0.676892668163,
                        'name': 'BOFUM',
                        'measurement_lost' : 8
                      }

    # keep motion options
    conditional_bofum_options = {
                        'omega': 0.0263632138087,
                        'extent': 7,
                        'noise_var': 0.743921945138,
                        'name': 'BOFMP with moving average',
                        'measurement_lost': 8,
                        'keep_motion': True,
                        'window_size': 4,
                        'initial_motion_factor': 0.562946012048,
                        'keep_motion_factor': 0.706594184009
                      }

    # BOFMP
    conditional_bofum_options = {
        'omega': 0.191234382275,
        'extent': 5,
        'noise_var': 0.644828151836,
        'name': 'BOFMP'}

    metrics = ['x_ent', 'f1_score', 'average_precision']

    result_folder_name = '05_{}_simluated_scenes_{}_num_steps_{}_model_{}_keep_motion'.format(simulated_scene, num_steps, cnn_model_name, return_flag)

    evaluations = []

    naive_bofum_evaluation = BofumEvaluationRealdata(scenes, num_steps,
                                                  naiveBOFUM, naive_bofum_options, metrics,
                                                  cnn_model=None,
                                                  cache_folder=result_folder_name,
                                                  simulated_scenes=simulated_scene)

    conditioanl_bofum_evaluation = BofumEvaluationRealdata(scenes, num_steps,
                                                  conditionalBOFUM, conditional_bofum_options, metrics,
                                                  cnn_model=cnn_model_name,
                                                  cache_folder=result_folder_name,
                                                  simulated_scenes=simulated_scene)

    evaluations.append(naive_bofum_evaluation)
    evaluations.append(conditioanl_bofum_evaluation)
    show_results(evaluations, metric='average_precision')