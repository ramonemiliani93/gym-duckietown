import math

import numpy as np

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.8
ANGLE_GAIN = 5
DISTANCE_GAIN = 5


class UAPD:
    def __init__(self, env, ref_velocity=REF_VELOCITY, angle_gain=ANGLE_GAIN, distance_gain=DISTANCE_GAIN):
        self.env = env
        self.ref_velocity = ref_velocity
        self.angle_gain = angle_gain
        self.distance_gain = distance_gain

    def predict(self, observation, metadata):

        # Return the angular velocity in order to control the Duckiebot so that it follows the lane.
        # Parameters:
        #     dist: distance from the center of the lane. Left is negative, right is positive.
        #     angle: angle from the lane direction, in rad. Left is negative, right is positive.
        # Outputs:
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.

        omega = 0.
        lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        dist = lane_pose.dist  # Distance to lane center. Left is negative, right is positive.
        angle = lane_pose.angle_rad % (2 * np.pi)

        # Threshold values
        angle_limit = np.pi / 6
        distance_limit = np.abs(self.angle_gain * angle_limit / self.distance_gain)
        # Restrict angle and distance
        angle_threshold = np.clip(angle, -angle_limit, angle_limit)
        dist_threshold = np.clip(dist, -distance_limit, distance_limit)

        # Add terms to control
        omega += self.angle_gain * angle_threshold
        omega += ((self.angle_gain ** 2) / (4 * self.ref_velocity)) * dist_threshold

        return [self.ref_velocity, omega], 0