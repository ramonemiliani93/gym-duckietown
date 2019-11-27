import ast

from ..algorithms.iil_testing import InteractiveImitationTesting
from ..learners import NeuralNetworkPolicy
from ..training._behaviors import Icra2019TestBehavior
from ..training._loggers import Logger
from ..training._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES
from ..training._parametrization import *
from ..training._settings import *


def test(config, entry):
    policy_parametrization = parametrization(
        iteration=config.parametrization,
        extra_parameters={'samples': 25, 'dropout': 0.9}
    )

    learner = NeuralNetworkPolicy(
        model=policy_parametrization,
        optimizer=None,
        storage_location=None,
        batch_size=32,
        epochs=50,
        model_path='/Users/ramon/Documents/udem/2-year/fall/gym-duckietown/icra2019/dagger/0/h1024e8/pure_pursuit_montecarlo_resnet/adam_lr_[0.001]/decay_0.5/model.pt'
    )

    return InteractiveImitationTesting(
        env=environment,
        teacher=teacher(environment),
        learner=learner,
        horizon=HORIZONS[config.test_horizon],
        episodes=EPISODES[config.test_horizon]
    )


if __name__ == '__main__':
    parser = process_args()
    parser.add_argument('--test-horizon', '-th', default=0, type=int)
    config = parser.parse_args()

    # training
    environment = simulation(at=MAP_STARTING_POSES[config.iteration])

    #
    logging_entry = experimental_entry(
        algorithm=ALGORITHMS[config.algorithm],
        experiment_iteration=config.iteration,
        parametrization_name=PARAMETRIZATIONS_NAMES[config.parametrization],
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
        learning_rate=LEARNING_RATES[config.learning_rate],
        # metadata=ast.literal_eval(config.metadata)
    )

    testing = test(config, entry=logging_entry)

    # observers
    driver = Icra2019TestBehavior(
        env=environment,
        starting_positions=MAP_STARTING_POSES,
        routine=testing
    )

    logger = Logger(
        env=environment,
        routine=testing,
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        log_file=logging_entry + 'testing.log'
    )

    testing.test(debug=True)

    environment.close()