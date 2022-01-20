import numpy as np
from SandboxSafety.NavUtils import pure_pursuit_utils
from SandboxSafety.NavUtils.speed_utils import calculate_speed, calculate_speed_fs0
import csv 

from SandboxSafety.NavAgents.TrackPP import PurePursuit as TrackPP

from matplotlib import pyplot as plt
import os, shutil

     
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret


def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return np.array(ret)



class RandomPlanner:
    def __init__(self, name="RandoPlanner"):
        self.d_max = 0.4 # radians  
        self.v = 2        
        self.name = name
        
        path = os.getcwd() + "/Data/Vehicles/" + self.name 
        # path = os.getcwd() + "/EvalVehicles/" + self.name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path
        np.random.seed(1)

    def plan_act(self, obs):
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        # Add random velocity
        return np.array([steering, self.v])

class ConstantPlanner:
    def __init__(self, name="StraightPlanner", value=0):
        self.steering_value = value
        self.v = 2        
        self.name = name

        path = os.getcwd() + "/PaperData/Vehicles/" + self.name 
        # path = os.getcwd() + "/EvalVehicles/" + self.name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path


    def plan_act(self, obs):
        return np.array([self.steering_value, self.v])



class PurePursuit:
    def __init__(self, sim_conf):
        self.name = "PurePursuit Planner"
        self.v = 6
        self.d_max= sim_conf.max_steer
        self.L = sim_conf.l_f + sim_conf.l_r
        self.lookahead_distance = 1
 
    def plan_act(self, obs):
        state = obs['state']
        pose_theta = state[2]
        x_follow = 2 # @x_follow is the param to change
        lookahead = np.array([x_follow, state[1]+self.lookahead_distance]) #pt 1 m in the future on centerline
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead[0:2]-state[0:2])
        if np.abs(waypoint_y) < 1e-6:
            return np.array([0, self.v])
        radius = 1/(2.0*waypoint_y/self.lookahead_distance**2)
        steering_angle = np.arctan(self.L/radius)
        steering_angle = np.clip(steering_angle, -self.d_max, self.d_max)

        v = min(self.v, calculate_speed_fs0(steering_angle))
        return np.array([steering_angle, v])





class RandoKernel:
    def construct_kernel(self, img_shape, obs_pts):
        pass

class EmptyPlanner:
    def __init__(self, planner, sim_conf):
        self.planner = planner
        self.kernel = RandoKernel()

    def plan(self, obs):
        return self.planner.plan_act(obs)
    

