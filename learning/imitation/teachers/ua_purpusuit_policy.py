import math
import numpy as np
from src.gym_duckietown.simulator import AGENT_SAFETY_RAD

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
GAIN = 10
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15


class UAPurePursuitPolicy:
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold

    def predict(self, observation, metadata):
        closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        if closest_point is None or closest_tangent is None:
            self.env.reset()
            closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        current_world_objects = self.env.objects
        velocity_slow_down = 1
        for obj in current_world_objects:
            if not obj.static and obj.kind == "duckiebot": 
                #TODO need to check the other object direction if he is moving in my same lane or not
                if abs(obj.pos[0] - self.env.cur_pos[0]) <0.1:
                    collision_penalty =  abs(obj.proximity(self.env.cur_pos, AGENT_SAFETY_RAD * AGENT_SAFETY_GAIN))
                    if collision_penalty > 0 :
                        # this means we are approaching and we need to slow down
                        velocity_slow_down = collision_penalty
                        break
                else:
                    pass

        lookup_distance = self.following_distance
        projected_angle, _, _= self._get_projected_angle_difference(0.3)
        scale = 1

        current_tile_pos = self.env.get_grid_coords(self.env.cur_pos)
        current_tile = self.env._get_tile(*current_tile_pos)
        if 'curve' in current_tile['kind'] or abs(projected_angle) < 0.92:
            # we have a corner brace yourselves
            scale = 0.5  
        _, closest_point, curve_point= self._get_projected_angle_difference(lookup_distance)

        if closest_point is None:  # if cannot find a curve point in max iterations
            return None, np.inf

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)
        right_vec = np.array([math.sin(self.env.cur_angle),0,math.cos(self.env.cur_angle)])
        dot = np.dot(right_vec, point_vec)
        omega = -1 * dot
        # range of dot is just -pi/2 and pi/2
        velocity = self.ref_velocity # * scale 
        if velocity_slow_down<1:
            velocity *= velocity_slow_down

        action = [velocity , omega]

        return action, 0.0
    

    def _get_projected_angle_difference(self, lookup_distance):
        # Find the projection along the path
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        curve_angle = None

        while iterations < 10:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_angle = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_angle is not None and curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_angle is None:  # if cannot find a curve point in max iterations
            return None, None, None

        else:
            return np.dot(curve_angle, closest_tangent), closest_point, curve_point
