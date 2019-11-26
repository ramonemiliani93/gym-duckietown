# learner's parametrization
from ..uncertainty_models import MonteCarloSqueezenet

PARAMETRIZATIONS_NAMES = ['montecarlo_squeezenet']
PARAMETRIZATIONS = [MonteCarloSqueezenet]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)
