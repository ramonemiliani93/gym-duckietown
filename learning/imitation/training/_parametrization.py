# learner's parametrization
from ..uncertainty_models import MonteCarloDronet

PARAMETRIZATIONS_NAMES = ['montecarlo_dronet']
PARAMETRIZATIONS = [MonteCarloDronet]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)
