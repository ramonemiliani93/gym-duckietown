import numpy as np
from .learner import BaseLearner


class RandomExploration(BaseLearner):

    def __init__(self, env, fake_uncertainty=1):
        self.uncertainty = fake_uncertainty

    def predict(self, observation, metadata):
        return np.random.uniform(0, 1, 2), np.random.uniform(0, 1, 2)

    def optimize(self, observations, action, episode):
        pass

    def save(self):
        print('I didn\'t learn a thing...')
