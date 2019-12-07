import math

from .dagger import DAgger
from .iil_learning import InteractiveImitationLearning
import numpy as np

class SimulatedDagger(DAgger):

    def __init__(self, env, teacher, learner, horizon, episodes):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes)
        # expert decay
        self.convergence_distance = 0.1
        self.convergence_angle = np.pi / 12 
        self.angle_limit = np.pi / 6
        self.distance_limit = 0.15

    def _mix(self):
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except :
            # this means we are need the teacher definitely
            return self.teacher
        if self.active_policy:
            # check for convergence if we are using the teacher to move back to our learner
            if abs(lp.dist) < self.convergence_distance and abs(lp.angle_rad)< self.convergence_angle:
                return self.learner
            else:
                return self.teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give 
            # control back to expert 
            if abs(lp.dist)> self.distance_limit or abs(lp.angle_rad) > self.angle_limit:
                return self.teacher
            else:
                return self.learner

    def _on_episode_done(self):
        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)
