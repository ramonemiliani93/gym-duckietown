# learner's parametrization
from ..uncertainty_models import MonteCarloResnet, IndividualMLP

PARAMETRIZATIONS_NAMES = ['feature', 'mc']
PARAMETRIZATIONS = [IndividualMLP, MonteCarloResnet]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)
