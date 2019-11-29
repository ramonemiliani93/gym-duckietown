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
        self.max_distance_limit = 0.48
        self.angle_limit = np.pi / 6
        self.distance_limit = 0.3
        self.max_n_episodes = 5# self._episodes // 4
        self.convergence_distance = 0.1
        self.convergence_angle = np.pi / 12 

    def _mix(self):
        # TODO create a better check for being within the road limits
        #TODO let the expert take over untill the bot converges back to the right position and then let the learner explore 
            
        use_teacher=False
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
            if abs(lp.dist)>self.distance_limit or abs(lp.angle_rad)> self.angle_limit:
                use_teacher=True
        except :
            use_teacher=True
        if self.active_policy:
            # check for convergence if we are using the teacher to move back to our learner
            if abs(lp.dist) < self.convergence_distance or abs(lp.angle_rad) < self.convergence_angle:
                use_teacher = False
            
        if use_teacher:
            # teacher needs to take over 
            return self.teacher
        else:
            # if we are in a good shape then we should explore more
            return self.learner

    def _on_episode_done(self):
        InteractiveImitationLearning._on_episode_done(self)
