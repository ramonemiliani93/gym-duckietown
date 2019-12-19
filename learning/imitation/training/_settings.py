import argparse
import math
import os

import numpy as np
import tensorflow as tf

from ..teachers import *

SEED = 19048  # generated by Google Random Generator (1 - 50,000)

MAX_VELOCITY = 0.8
if 'VISUALIZE' in os.environ:
    DEBUG = True
else:
    DEBUG = False

# seeding
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAP_NAME = 'udem1' #'loop_pedestrians'#'loop_dyn_lfv' #loop_empty
MAP_STARTING_POSES = [
    [[0.8, 0.0, 1.5], 10.90],
    [[0.8, 0.0, 2.5], 10.90],
    [[1.5, 0.0, 3.5], 12.56],
    [[2.5, 0.0, 3.5], 12.56],
    [[4.1, 0.0, 2.0], 14.14],
    [[2.8, 0.0, 0.8], 15.71],
]

# all with Dataset Aggregation
ALGORITHMS = ['supervised', 'dagger', 'aggrevate', 'dropout_dagger', 'upms', 'upms-ne', 'upms-sl', 'upms-ne-sl']

# teacher
teacher_name = 'pure_pursuit'

# Task Configuration
HORIZONS = [128, 512, 512, 1024, 2048]
EPISODES = [200, 32, 16, 8, 8]
# decays
MIXING_DECAYS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# uncertainty threshold
UNCERTAINTY_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
ITERATIONS = 4  # to 4
SUBMISSION_DIRECTORY = 'dt2019'


def experimental_entry(algorithm, experiment_iteration, parametrization_name, horizon, episodes,
                       optimization_name, learning_rate, metadata=None):
    entry = '{}/{}/{}/h{}e{}/{}_{}/{}_lr_{}/'.format(
        SUBMISSION_DIRECTORY,
        algorithm,
        experiment_iteration,
        horizon,
        episodes,
        teacher_name,
        parametrization_name,
        optimization_name,
        learning_rate
    )

    if metadata is not None:
        for key in metadata:
            entry += '{}_{}/'.format(key, metadata[key])

    return entry


def simulation(at, env=None, reset=True, is_testing=False):
    from src.gym_duckietown.envs import DuckietownEnv
    if env is None:
        if not(is_testing):
            environment = DuckietownEnv(
                domain_rand=True,
                max_steps=math.inf,
                map_name=MAP_NAME,
                randomize_maps_on_reset=False,
                randomize_map_parent_dir='lf'
            )
        else:
            environment = DuckietownEnv(
                domain_rand=False,
                max_steps=math.inf,
                map_name=MAP_NAME,
                randomize_maps_on_reset=True,
                randomize_map_parent_dir='lf'
            )
    else:
        environment = env

    if reset:
        environment.reset()

    environment.cur_pos = np.array(at[0])
    environment.cur_angle = at[1]

    return environment


def robot():
    from src.gym_duckietown.envs import DuckiebotEnv
    return DuckiebotEnv()


def teacher(env):
    return UAPurePursuitPolicy(
        env=env,
        ref_velocity=MAX_VELOCITY
    )

# def teacher(env):
#     return Stanley(
#         env=env,
#         max_velocity=MAX_VELOCITY
#     )

# def teacher(env):
#     return StanleyLFV(
#         env=env
#     )

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', '-a', default=1, type=int)
    parser.add_argument('--iteration', '-i', default=0, type=int)
    parser.add_argument('--horizon', '-r', default=0, type=int)
    parser.add_argument('--parametrization', '-p', default=0, type=int)
    parser.add_argument('--optimization', '-o', default=5, type=int)
    parser.add_argument('--learning-rate', '-l', default=2, type=int)
    parser.add_argument('--metadata', '-m', default=None)

    return parser
