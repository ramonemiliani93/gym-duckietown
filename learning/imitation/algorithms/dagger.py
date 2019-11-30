import numpy as np

from .iil_learning import InteractiveImitationLearning


class DAgger(InteractiveImitationLearning):

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.5):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes)
        # expert decay
        self.p = alpha
        self.alpha = self.p
        self.convergence_distance = 0.1
        self.convergence_angle = np.pi / 12 
        self.angle_limit = np.pi / 6 
        self.distance_limit = 0.2

    def _mix(self):
        control_policy = np.random.choice(
            a=[self.teacher, self.learner],
            p=[self.alpha, 1. - self.alpha]
        )
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except :
            # this means we are need the teacher definitely
            return self.teacher
        if self.active_policy:
            # check for convergence if we are using the teacher to move back to our learner
            if abs(lp.dist) < self.convergence_distance:
                return self.learner
        else:
            # in case we are using our learner and it started to diverge a lot we need to give 
            # control back to expert 
            if abs(lp.dist)>self.distance_limit or abs(lp.angle_rad)> self.angle_limit:
                return self.teacher


        return control_policy

    def _on_episode_done(self):
        # decay expert probability after each episode
        self.alpha = self.p ** self._episode

        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)
