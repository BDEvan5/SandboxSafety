import numpy as np
from SandboxSafety.NavUtils import pure_pursuit_utils
from SandboxSafety.NavUtils.speed_utils import calculate_speed
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
        self.v = 2
        self.d_max= sim_conf.max_steer
        self.L = sim_conf.l_f + sim_conf.l_r
        self.lookahead_distance = 1
 
    def plan_act(self, obs):
        state = obs['state']
        pose_theta = state[2]
        lookahead = np.array([1, state[1]+self.lookahead_distance]) #pt 1 m in the future on centerline
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead[0:2]-state[0:2])
        if np.abs(waypoint_y) < 1e-6:
            return np.array([0, self.v])
        radius = 1/(2.0*waypoint_y/self.lookahead_distance**2)
        steering_angle = np.arctan(self.L/radius)
        steering_angle = np.clip(steering_angle, -self.d_max, self.d_max)
        return np.array([steering_angle, self.v])



class TrackPurePursuit:
    def __init__(self, sim_conf) -> None:
        self.name = "Track Pure Pursuit"

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.max_steer = sim_conf.max_steer

        self.v_gain = 0.5
        self.lookahead = 0.8
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None

        self._load_csv_track(sim_conf.map_name)

    def _get_current_waypoint(self, position):
        lookahead_distance = self.lookahead
    
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.vs[i])
        else:
            return None

    def plan_act(self, obs):
        pose_th = obs['state'][2]
        pos = np.array(obs['state'][0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)


        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # speed = 4
        speed = calculate_speed(steering_angle)

        return [steering_angle, speed]

    def _load_csv_track(self, map_name):
        track = []
        filename = 'maps/' + map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5]


        # plt.figure(1)
        # plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], '+-', markersize=10)
        # plt.show()

        self.expand_wpts()

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.waypoints = np.array(new_line)
        self.vs = np.array(new_vs)



class RandoKernel:
    def construct_kernel(self, img_shape, obs_pts):
        pass

class EmptyPlanner:
    def __init__(self, planner, sim_conf):
        self.planner = planner
        self.kernel = RandoKernel()

    def plan(self, obs):
        return self.planner.plan_act(obs)
    

