from ..algorithms import DAgger, SimulatedDagger
from ..learners import NeuralNetworkPolicy
from ..training._behaviors import Icra2019Behavior
from ..training._loggers import IILTrainingLogger
from ..training._optimization import *
from ..training._parametrization import *
from ..training._settings import *
from learning.imitation.learners import RandomExploration

MIXING_DECAYS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def dagger(env, teacher, experiment_iteration, selected_parametrization, selected_optimization, selected_learning_rate,
           selected_horizon, selected_episode, selected_mixing_decay):
    task_horizon = HORIZONS[selected_horizon]
    task_episodes = EPISODES[selected_episode]

    policy_parametrization = parametrization(
        iteration=selected_parametrization,
        extra_parameters={'p': 0.2, 'num_outputs': 2, 'num_samples': 1}
    )

    policy_optimizer = optimizer(
        optimizer_iteration=selected_optimization,
        learning_rate_iteration=selected_learning_rate,
        parametrization=policy_parametrization,
        task_metadata=[task_horizon, task_episodes, 1]
    )
    # learner = RandomExploration(env)
    learner = NeuralNetworkPolicy(
        model=policy_parametrization,
        optimizer=policy_optimizer,
        storage_location=experimental_entry(
            algorithm='dagger',
            experiment_iteration=config.iteration,
            parametrization_name=PARAMETRIZATIONS_NAMES[config.parametrization],
            horizon=HORIZONS[config.horizon],
            episodes=EPISODES[config.horizon],
            optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
            learning_rate=LEARNING_RATES[config.learning_rate],
            metadata={
                'decay': MIXING_DECAYS[config.decay]
            }
        ),
        batch_size=16,
        epochs=50,
        input_shape=(120,160),
        max_velocity = MAX_VELOCITY
    )

    return SimulatedDagger(env=env,
                        teacher=teacher,
                        learner=learner,
                        horizon = task_horizon,
                        episodes=task_episodes,
                        alpha=MIXING_DECAYS[config.decay])



if __name__ == '__main__':
    parser = process_args()
    parser.add_argument('--decay', '-d', default=6, type=int)

    config = parser.parse_args()
    # training
    environment = simulation(at=MAP_STARTING_POSES[config.iteration])

    algorithm = dagger(
        env=environment,
        teacher=teacher(environment),
        experiment_iteration=config.iteration,
        selected_parametrization=config.parametrization,
        selected_optimization=config.optimization,
        selected_horizon=config.horizon,
        selected_episode=config.horizon,
        selected_learning_rate=config.learning_rate,
        selected_mixing_decay=config.decay
    )

    # observers
    driver = Icra2019Behavior(
        env=environment,
        at=MAP_STARTING_POSES[config.iteration],
        routine=algorithm
    )

    disk_entry = experimental_entry(
        algorithm='dagger',
        experiment_iteration=config.iteration,
        parametrization_name='test', #PARAMETRIZATIONS_NAMES[config.parametrization],
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
        learning_rate=LEARNING_RATES[config.learning_rate],
        metadata={
            'decay': MIXING_DECAYS[config.decay]
        }
    )
    logs = IILTrainingLogger(
        env=environment,
        routine=algorithm,
        log_file=disk_entry + 'training.log',
        data_file=disk_entry + 'dataset_evolution.pkl',
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon]
    )

    algorithm.train(debug=False)  #DEBUG

    environment.close()
