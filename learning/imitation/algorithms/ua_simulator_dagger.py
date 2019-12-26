import math

from .dagger import DAgger
from .iil_learning import InteractiveImitationLearning
import numpy as np

class SimulatedDagger(DAgger):

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.5):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes)
        # expert decay
        self.p = alpha
        self.alpha = self.p
        self.convergence_distance = 0.05
        self.convergence_angle = np.pi / 18 
        # making this limit stricter to avoid full spin as sometimes get lp values might get u the closer other direction lane if ur current angle is high 
        self.angle_limit = np.pi / 8
        self.distance_limit = 0.15

    def _mix(self):
        control_policy = np.random.choice(
            a=[self.teacher, self.learner],
            p=[self.alpha, 1. - self.alpha]
        )
        if self._found_obstacle:
            return self.teacher
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except :
            return control_policy
        if self.active_policy:
            # check for convergence if we are using the teacher to move back to our learner
            if not(abs(lp.dist) < self.convergence_distance and abs(lp.angle_rad)< self.convergence_angle):
                return self.teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give 
            # control back to expert 
            if abs(lp.dist)> self.distance_limit or abs(lp.angle_rad) > self.angle_limit:
                return self.teacher
        return control_policy

    def _on_episode_done(self):
        self.alpha = self.p ** self._episode
        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)
