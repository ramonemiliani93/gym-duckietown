# learner's parametrization
from ..uncertainty_models import MonteCarloResnet, IndividualMLP

PARAMETRIZATIONS_NAMES = ['montecarlo_resnet', 'feature']
PARAMETRIZATIONS = [MonteCarloResnet, IndividualMLP]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)
