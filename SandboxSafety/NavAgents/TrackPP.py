from SandboxSafety.NavUtils.Trajectory import Trajectory
import numpy as np 
from matplotlib import pyplot as plt
from SandboxSafety.NavUtils import pure_pursuit_utils as pp_utils


class PurePursuit:
    def __init__(self, conf):
        self.name = "PurePursuit"
        

        self.trajectory = Trajectory(conf.map_name)

        self.lookahead = conf.lookahead
        self.vgain = conf.v_gain
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer

        self.progresses = []
        self.aim_pts = []

    def plan_act(self, obs):
        state = obs['state']
        pose_th = state[2]
        pos = state[0:2]
        v_current = state[3]

        self.progresses.append(obs['target'][1])

        v_min_plan = 1
        if v_current < v_min_plan:
            return np.array([0, 7])

        lookahead_point = self.trajectory.get_current_waypoint(pos, self.lookahead)
        self.aim_pts.append(lookahead_point)

        speed, steering_angle = pp_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        speed *= self.vgain

        # speed = calculate_speed(steering_angle)

        return np.array([steering_angle, speed])

    def plot_progress(self):
        plt.figure(2)
        plt.clf()
        # plt.plot(self.progresses)
        aim_pts = np.array(self.aim_pts)
        plt.plot(aim_pts[:, 0], aim_pts[:, 1])

        # plt.show()
        plt.pause(0.0001)

        # self.aim_pts.clear()
