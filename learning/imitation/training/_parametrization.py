# learner's parametrization
from ..uncertainty_models import MonteCarloMnasnet

PARAMETRIZATIONS_NAMES = ['montecarlo_mnasnet']
PARAMETRIZATIONS = [MonteCarloMnasnet]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)
