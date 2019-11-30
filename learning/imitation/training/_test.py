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
        extra_parameters={'samples': 1, 'dropout': 0.9, 'num_outputs': 2}
    )

    learner = NeuralNetworkPolicy(
        model=policy_parametrization,
        optimizer=None,
        storage_location=None,
        batch_size=16,
        epochs=50,
        model_path='model_path',
        input_shape = (120,160)
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