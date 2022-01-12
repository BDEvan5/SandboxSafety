from os import name
import numpy as np 
import csv, torch
from matplotlib import pyplot as plt

from SandboxSafety.NavUtils.TD3 import TD3
from SandboxSafety import LibFunctions as lib
from SandboxSafety.NavUtils.HistoryStructs import TrainHistory
from SandboxSafety.NavUtils.speed_utils import calculate_speed
from SandboxSafety.NavUtils import pure_pursuit_utils
from SandboxSafety.NavUtils.RewardFunctions import DistReward, CthReward

class SerialPP:
    def __init__(self, sim_conf) -> None:
        self.path_name = None

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.max_steer = sim_conf.max_steer

        self.v_gain = 0.5
        self.lookahead = 0.8
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None

        self.aim_pts = []

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

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        self.aim_pts.append(lookahead_point[0:2])

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # speed = 4
        speed = calculate_speed(steering_angle)

        return [steering_angle, speed]

    def reset_lap(self):
        self.aim_pts.clear()


class ModHistory:
    def __init__(self) -> None:
        self.mod_history = []
        self.pp_history = []
        self.reward_history = []
        self.critic_history = []

    def add_step(self, pp, nn, c_val=None):
        self.pp_history.append(pp)
        self.mod_history.append(nn)
        self.critic_history.append(c_val)

    def save_nn_output(self):
        # save_csv_data(self.critic_history, 'Vehicles/nn_output.csv')
        plt.figure(5)
        plt.plot(self.mod_history)
        plt.pause(0.0001)
        plt.figure(6)
        plt.plot(self.pp_history)
        plt.pause(0.0001)

        self.mod_history.clear()
        self.pp_history.clear()
        self.critic_history.clear()


class SerialBase(SerialPP):
    def __init__(self, agent_name, sim_conf) -> None:
        super().__init__(sim_conf)
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.range_finder_scale = sim_conf.range_finder_scale 

        self.history = ModHistory()

        self._load_csv_track(sim_conf.map_name)

    def transform_obs(self, obs, pp_action):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env
            pp_action: [steer, speed] from pure pursuit controller

        Returns:
            nn_obs: observation vector for neural network
        """
        state = obs['state']
        scan = obs['scan']/ self.range_finder_scale
        target = obs['target']

        cur_v = [state[3]/self.max_v]
        cur_d = [state[4]/self.max_steer]
        target_angle = [target[0]/self.max_steer]
        dr_scale = [pp_action[0]/self.max_steer]

        # nn_obs = np.concatenate([cur_v, cur_d, target_angle, scan])
        nn_obs = np.concatenate([cur_v, cur_d, target_angle, dr_scale, scan])

        return nn_obs

    def modify_references(self, nn_action, d_ref):
        """
        Modifies the reference quantities for the steering.
        Mutliplies the nn_action with the max steering and then sums with the reference

        Args:
            nn_action: action from neural network in range [-1, 1]
            d_ref: steering reference from PP

        Returns:
            d_new: modified steering reference
        """
        d_new = self.max_steer * nn_action[0] 
        d_new = np.clip(d_new, -self.max_steer, self.max_steer)

        return d_new

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

        self.expand_wpts()

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.waypoints = np.array(new_line)
        self.vs = np.array(new_vs)


class SerialVehicleTrain(SerialBase):
    def __init__(self, agent_name, sim_conf, load=False):
        """
        Training vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
            load: if the network should be loaded or recreated.
        """

        SerialBase.__init__(self, agent_name, sim_conf)

        self.path = 'EvalVehicles/' + agent_name
        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.try_load(load, sim_conf.h_size, self.path)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.t_his = TrainHistory(agent_name, load)

        # self.calculate_reward = DistReward() 
        self.calculate_reward = CthReward(0.004, 0.004) 


    def plan_act(self, obs):
        pp_action = super().act_pp(obs['state'])
        nn_obs = self.transform_obs(obs, pp_action)
        self.add_memory_entry(obs, nn_obs)

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        self.nn_act = nn_action

        self.nn_state = nn_obs

        steering_angle = self.modify_references(self.nn_act, pp_action[0])
        speed = calculate_speed(steering_angle)
        self.action = np.array([steering_angle, speed])

        return self.action

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = self.calculate_reward(self.state, s_prime)

            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        pp_action = super().act_pp(s_prime['state'])
        nn_s_prime = self.transform_obs(s_prime, pp_action)
        reward = self.calculate_reward(self.state, s_prime)


        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
            self.agent.save(self.path)
        self.state = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)


class SerialVehicleTest(SerialBase):
    def __init__(self, agent_name, sim_conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        SerialBase.__init__(self, agent_name, sim_conf)

        self.path = 'EvalVehicles/' + agent_name
        self.actor = torch.load(self.path + '/' + agent_name + "_actor.pth")
        self.n_beams = 10

        print(f"Agent loaded: {agent_name}")

    def plan_act(self, obs):
        pp_action = super().act_pp(obs['state'])
        nn_obs = self.transform_obs(obs, pp_action)

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        # critic_val = self.agent.get_critic_value(nn_obs, nn_action)
        # self.history.add_step(pp_action[0], nn_action[0]*self.max_steer, critic_val)

        steering_angle = self.modify_references(self.nn_act, pp_action[0])
        speed = calculate_speed(steering_angle)
        action = np.array([steering_angle, speed])

        return action
