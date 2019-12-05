import math

import numpy as np
from src.gym_duckietown.simulator import NotInLane
POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.8
GAIN = 1.5
FOLLOWING_DISTANCE = 0.35


class Stanley:
    def __init__(self, env, ref_velocity=REF_VELOCITY, gain=GAIN, following_distance=FOLLOWING_DISTANCE, position_threshold=POSITION_THRESHOLD):
        self.env = env
        self.ref_velocity = ref_velocity
        self.gain = gain
        self.following_distance = following_distance
        self.position_threshold = position_threshold
        self.max_speed = 0.8
        self.min_speed = 0.25

    def predict(self, observation, metadata):

        # Return the angular velocity in order to control the Duckiebot so that it follows the lane.
        # Parameters:
        #     dist: distance from the center of the lane. Left is negative, right is positive.
        #     angle: angle from the lane direction, in rad. Left is negative, right is positive.
        # Outputs:
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.

        steering_angle = 0.
        try:
            lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        except NotInLane:
            self.env.reset()
            lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        dist = lane_pose.dist  # Distance to lane center. Left is negative, right is positive.
        angle = lane_pose.angle_rad 

        # Project to curve to find curvature
        projected_angle_difference, closest_point = self._get_projected_angle_difference()
        # fixing velocity
        if projected_angle_difference>0.91:
            velocity = self.max_speed 
        else:
            velocity = self.min_speed 

        # Add terms to control
        steering_angle += angle
        steering_angle += np.arctan2(self.gain * dist, velocity)  

        # Translate to angular speed
        #TODO use the model to predict the steering angle which is easier to tune on the bot
        omega = velocity * np.sin(steering_angle) * 27 # v sin(theta) / r
        action = [velocity, omega]

        position_diff = np.linalg.norm(closest_point - self.env.cur_pos, ord=1)
        if position_diff > self.position_threshold:  # or velocity_diff > 0.5:
            return action, 0.0
        else:
            if metadata[0] == 0:
                return action, 0.0
            if metadata[1] is None:
                return action, 0.0

        return None, math.inf

    def _get_projected_angle_difference(self):
        # Find the projection along the path
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        lookup_distance = self.following_distance
        curve_angle = None

        while iterations < 10:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            _, curve_angle = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_angle is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_angle is None:  # if cannot find a curve point in max iterations
            return 0

        else:
            return np.dot(curve_angle, closest_tangent), closest_point