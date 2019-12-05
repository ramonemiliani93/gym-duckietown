import numpy as np

from .iil_learning import InteractiveImitationLearning


class DAgger(InteractiveImitationLearning):

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.5):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes)
        # expert decay
        self.p = alpha
        self.alpha = self.p

    def _mix(self):
        control_policy = np.random.choice(
            a=[self.teacher, self.learner],
            p=[self.alpha, 1. - self.alpha]
        )

        return control_policy

    def _on_episode_done(self):
        # decay expert probability after each episode
        self.alpha = self.p ** self._episode

        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)
