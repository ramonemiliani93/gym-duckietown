from abc import ABC, abstractmethod


class BaseLearner(ABC):

    @abstractmethod
    def optimize(self, observations, expert_actions, episode):
        pass

    @abstractmethod
    def predict(self, observation, metadata):
        pass

    @abstractmethod
    def save(self):
        pass