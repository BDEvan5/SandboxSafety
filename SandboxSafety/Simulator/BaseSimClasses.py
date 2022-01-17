import numpy as np 
from matplotlib import pyplot as plt
import os
from numba import njit

import LearningLocalPlanning.LibFunctions as lib
from SandboxSafety.Simulator.LaserScanner import ScanSimulator
from SandboxSafety.Simulator.Dynamics import update_complex_state, update_std_state


#TODO: move this to another location
class SimHistory:
    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []


        self.ctr = 0

    def save_history(self):
        pos = np.array(self.positions)
        vel = np.array(self.velocities)
        steer = np.array(self.steering)
        obs = np.array(self.obs_locations)

        d = np.concatenate([pos, vel[:, None], steer[:, None]], axis=-1)

        d_name = 'Vehicles/TrainData/' + f'data{self.ctr}'
        o_name = 'Vehicles/TrainData/' + f"obs{self.ctr}"
        np.save(d_name, d)
        np.save(o_name, obs)

    def reset_history(self):
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.ctr += 1

    def show_history(self, vs=None):
        plt.figure(1)
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        if vs is not None:
            r = len(vs) / len(self.velocities)
            new_vs = []
            for i in range(len(self.velocities)):
                new_vs.append(vs[int(round(r*i))])
            plt.plot(new_vs)
            plt.legend(['Actual', 'Planned'])
        plt.pause(0.001)

    def show_forces(self):
        mu = self.sim_conf['car']['mu']
        m = self.sim_conf['car']['m']
        g = self.sim_conf['car']['g']
        l_f = self.sim_conf['car']['l_f']
        l_r = self.sim_conf['car']['l_r']
        f_max = mu * m * g
        f_long_max = l_f / (l_r + l_f) * f_max

        self.velocities = np.array(self.velocities)
        self.thetas = np.array(self.thetas)

        # divide by time taken for change to get per second
        t = self.sim_conf['sim']['timestep'] * self.sim_conf['sim']['update_f']
        v_dot = (self.velocities[1:] - self.velocities[:-1]) / t
        oms = (self.thetas[1:] - self.thetas[:-1]) / t

        f_lat = oms * self.velocities[:-1] * m
        f_long = v_dot * m
        f_total = (f_lat**2 + f_long**2)**0.5

        plt.figure(3)
        plt.clf()
        plt.title("Forces (lat, long)")
        plt.plot(f_lat)
        plt.plot(f_long)
        plt.plot(f_total, linewidth=2)
        plt.legend(['Lat', 'Long', 'total'])
        plt.plot(np.ones_like(f_lat) * f_max, '--')
        plt.plot(np.ones_like(f_lat) * f_long_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_long_max, '--')
        plt.pause(0.001)


class BaseSim:
    """
    Base simulator class

    Important parameters:
        timestep: how long the simulation steps for
        max_steps: the maximum amount of steps the sim can take

    Data members:
        car: a model of a car with the ability to update the dynamics
        scan_sim: a simulator for a laser scanner
        action: the current action which has been given
        history: a data logger for the history
    """
    def __init__(self, env_map, done_fcn, sim_conf):
        """
        Init function

        Args:
            env_map: an env_map object which holds a map and has mapping functions
            done_fcn: a function which checks the state of the simulation for episode completeness
        """
        self.done_fcn = done_fcn
        self.env_map = env_map
        self.sim_conf = sim_conf #TODO: don't store the conf file, just use and throw away.
        self.n_obs = self.env_map.n_obs

        self.timestep = self.sim_conf.sim_time_step
        self.max_steps = self.sim_conf.max_steps
        self.plan_steps = self.sim_conf.update_steps
        self.n_beams = self.sim_conf.n_beams

        self.state = np.zeros(5)
        self.scan_sim = ScanSimulator(self.sim_conf.n_beams)
        self.scan_sim.init_sim_map(env_map)

        self.done = False
        self.colission = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.history = SimHistory(self.sim_conf)
        self.done_reason = ""

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v
        self.previous_progress = 0.000000

    def step_plan(self, action):
        """
        Takes multiple control steps based on the number of control steps per planning step

        Args:
            action: [steering, speed]
            done_fcn: a no arg function which checks if the simulation is complete
        """
        action = np.array(action)
        self.action = action

        self.state = update_complex_state(self.state, action, self.timestep)
        self.steps += 1 

        self.done_fcn()

        self.check_angle_limit()

        self.record_history(action)

        obs = self.get_observation()
        # angle_diff = lib.sub_angles_complex(obs['target'][2], self.state[2]) 
        # if abs(angle_diff) > 0.95*np.pi:
        #     self.reward = 0
        #     self.done = True 
        #     self.done_reason = f"Wrong direction: {self.state[2]},track direction: {obs['target'][2]} -> diff: {angle_diff}"
        #     print(f"{self.done_reason}")
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def check_angle_limit(self):
        """
        Checks if the angle is within the limits of the car
        """
        if self.state[2] > np.pi:
            self.state[2] -= 2*np.pi
        elif self.state[2] < -np.pi:
            self.state[2] += 2*np.pi

    def record_history(self, action):
        self.action = action
        self.history.velocities.append(self.state[3])
        self.history.steering.append(self.state[4])
        self.history.positions.append(self.state[0:2])
        self.history.thetas.append(self.state[2])

    def reset(self, add_obs=True):
        """
        Resets the simulation

        Args:
            add_obs: a boolean flag if obstacles should be added to the map

        Returns:
            state observation
        """
        self.done = False
        self.done_reason = "Null"
        self.action_memory = []
        self.steps = 0
        self.reward = 0
        self.previous_progress = 0

        self.state[0:3] = self.env_map.start_pose 
        self.state[3:5] = np.zeros(2)

        self.history.reset_history()

        if add_obs:
            self.env_map.add_obstacles()

        # update the dt img in the scan simulator after obstacles have been added
        dt = self.env_map.set_dt()
        self.scan_sim.dt = dt

        return self.get_observation()

    def fake_reset(self):
        """
        Resets the simulation

        Args:
            add_obs: a boolean flag if obstacles should be added to the map

        Returns:
            state observation
        """
        self.done = False
        self.done_reason = "Null"
        self.action_memory = []
        self.steps = 0
        self.reward = 0
        self.previous_progress = 0

        self.history.reset_history()

        return self.get_observation()

    def render(self, wait=False, name="No vehicle name set"):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)
        plt.title(name)

        xs, ys = self.env_map.convert_positions(self.history.positions)
        plt.plot(xs, ys, 'r', linewidth=3)
        plt.plot(xs, ys, '+', markersize=12)

        x, y = self.env_map.xy_to_row_column(self.state[0:2])
        plt.plot(x, y, 'x', markersize=20)

        text_x = self.env_map.map_width + 1
        text_y = self.env_map.map_height / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.state[0]:.2f}, {self.state[1]:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.state[3]:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.state[2] * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.state[4]*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def render_trajectory(self, path, save_name="VehicleName", safety_history=None):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)
        plt.title(save_name)

        xs, ys = self.env_map.convert_positions(self.history.positions)
        if safety_history is None:
            plt.plot(xs, ys, 'r', linewidth=3)
            # plt.plot(xs, ys, '+', markersize=12)
        else:
            N = len(safety_history.planned_actions)
            for i in range(N-1):
                x_pts = [xs[i], xs[i+1]]
                y_pts = [ys[i], ys[i+1]]
                if safety_history.planned_actions[i] == safety_history.safe_actions[i]:
                    plt.plot(x_pts, y_pts, 'r', linewidth=3)
                else:
                    plt.plot(x_pts, y_pts, 'b', linewidth=3)
                

        plt.savefig(f"{path}/{save_name}_track.svg")
        plt.savefig(f"{path}/{save_name}_track.png") # for easy viewing


    def get_target_obs(self):
        target = self.env_map.end_goal
        pos = self.state[0:2]
        base_angle = lib.get_bearing(pos, target) 
        angle = lib.sub_angles_complex(base_angle, self.state[2])

        em = self.env_map
        s, idx = calculate_track_progress_idx(pos, em.ref_pts, em.diffs, em.l2s, em.ss_normal)
        
        lower_idx = int(max(0, idx-2))
        upper_idx = int(min(len(em.ref_pts)-1, idx+2))
        track_angle = lib.get_bearing(em.ref_pts[lower_idx], em.ref_pts[upper_idx])

        return [angle, s, track_angle]
    
    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        car_obs = self.state
        pose = car_obs[0:3]
        scan = self.scan_sim.scan(pose,self.n_beams)
        target = self.get_target_obs()

        observation = {}
        observation['state'] = car_obs
        observation['scan'] = scan 
        observation['full_scan'] = self.scan_sim.scan(pose, 1000)
        observation['target'] = target
        observation['reward'] = self.reward

        return observation




@njit(cache=True)
def calculate_track_progress_idx(point, wpts, diffs, l2s, ss):
    dots = np.empty((wpts.shape[0]-1, ))
    dots_shape = dots.shape[0]
    for i in range(dots_shape):
        dots[i] = np.dot((point - wpts[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0  #np.clip, unsupported
    
    projections = wpts[:-1,:] + (t*diffs.T).T

    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))

    min_dist_segment = np.argmin(dists)
    dist_from_cur_pt = dists[min_dist_segment]

    s = ss[min_dist_segment] + dist_from_cur_pt
    # print(F"{min_dist_segment} --> SS: {ss[min_dist_segment]}, curr_pt: {dist_from_cur_pt}")

    s = s / ss[-1]

    return s, min_dist_segment


