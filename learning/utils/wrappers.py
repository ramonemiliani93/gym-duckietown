import gym
from gym import spaces
import numpy as np
import collections

# to simlate bluring happeninginour simulator
class MotionBlurWrapper(gym.ObservationWrapper):
    """
    Simulate the motion blur (unstatefully), simulated in the AIDO Challenge Interface
    """
    def __init__(self, env=None, window_size=20):
        from collections import deque
        gym.ObservationWrapper.__init__(self, env)
        self.n_images = 10
        self.observation_history =  collections.deque([], self.n_images)
    def observation(self, observation):  
        import cv2
        blured = None
        if len(self.observation_history)== self.n_images:
            # do the average trick
            blured = self.observation_history[0]
            for i in range(1, self.n_images):
                blured = cv2.addWeighted(blured,0.5,self.observation_history[i],0.5,0)
            
            return cv2.addWeighted(observation,0.6,blured,0.4,0)
        else:
            # do the blurring part
            size = 10
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            # applying the kernel to the input image
            blured = cv2.filter2D(observation, -1, kernel_motion_blur)
        self.observation_history.append(observation)
        return  blured

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from scipy.misc import imresize
        return imresize(observation, self.shape)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_
