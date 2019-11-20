# learner's parametrization
from ..uncertainty_models import MonteCarloResnet

PARAMETRIZATIONS_NAMES = ['montecarlo_resnet']
PARAMETRIZATIONS = [MonteCarloResnet]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)
