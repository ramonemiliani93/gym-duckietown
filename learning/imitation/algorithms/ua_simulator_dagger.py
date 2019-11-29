import math

from .dagger import DAgger
from .iil_learning import InteractiveImitationLearning
import numpy as np

class SimulatedDagger(DAgger):
    def __init__(self, env, teacher, learner, horizon, episodes):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, None)
        self.learner_simulated_uncertainty = None 
        # starting with strict limits to use the teacher more and then relaxing this limit to allow the model to explore
        self.max_angle_limit = np.pi/6 
        self.max_distance_limit = 0.35
        self.angle_limit = np.pi / 12
        self.distance_limit = 0.1
        self.max_n_episodes = self._episodes // 4

    def _mix(self):
        # TODO create a better check for being within the road limits
        use_teacher=False
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
            if abs(lp.dist)>self.distance_limit:
                use_teacher=True
            if abs(lp.angle_rad)> self.angle_limit:
                use_teacher=True
        except :
            use_teacher=True
        if use_teacher:
            # teacher needs to take over
            return self.teacher
        else:
            # if we are in a good shape then we should explore more
            return self.learner

    def _on_episode_done(self):
        decay = max(0.0,(self._episode*1.0/self.max_n_episodes))
        self.angle_limit = self.max_angle_limit * decay
        self.distance_limit = self.max_distance_limit * decay
        InteractiveImitationLearning._on_episode_done(self)
