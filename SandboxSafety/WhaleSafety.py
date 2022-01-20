
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import yaml, csv
from SandboxSafety.Simulator.Dynamics import update_complex_state, update_std_state
from SandboxSafety.Modes import Modes

class SafetyHistory:
    def __init__(self):
        self.planned_actions = []
        self.safe_actions = []

    def add_locations(self, planned_action, safe_action=None):
        self.planned_actions.append(planned_action)
        if safe_action is None:
            self.safe_actions.append(planned_action)
        else:
            self.safe_actions.append(safe_action)

    def plot_safe_history(self):
        plt.figure(5)
        plt.clf()
        plt.title("Safe History")
        plt.plot(self.planned_actions, color='blue')
        plt.plot(self.safe_actions, '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        self.planned_actions = []
        self.safe_actions = []

    def save_safe_history(self, path, name):
        plt.figure(5)
        plt.clf()
        plt.title(f"Safe History: {name}")
        plt.plot(self.planned_actions, color='blue')
        plt.plot(self.safe_actions, color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        plt.savefig(f"{path}/{name}_actions.png")

        data = []
        for i in range(len(self.planned_actions)):
            data.append([i, self.planned_actions[i], self.safe_actions[i]])
        full_name = path + f'/{name}_training_data.csv'
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)


        self.planned_actions = []
        self.safe_actions = []

verbose = False
# verbose = True


class LobsterSupervisor:
    def __init__(self, planner, kernel, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        self.d_max = conf.max_steer
        # self.v = 2
        self.kernel = kernel
        self.planner = planner
        self.safe_history = SafetyHistory()
        self.intervene = False

        self.time_step = conf.lookahead_time_step

        # aliases for the test functions
        try:
            self.n_beams = planner.n_beams
        except: pass
        self.plan_act = self.plan
        self.name = planner.name

        self.m = Modes(conf)

    def plan(self, obs):
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])

        if not self.kernel.check_state(state).any():
            inds = self.kernel.get_indices(state)
            print(f"Current state UNSAFE -> Kernel inds: {inds}")
            # np.save(f"temp_kernel_for_inds.npy", self.kernel.kernel)
            return [0, 2]

        init_mode_action, id = self.action2mode(init_action)
        safe, next_state = self.check_init_action(state, init_mode_action)
        if safe:
            self.safe_history.add_locations(init_mode_action[0], init_mode_action[0])
            return init_mode_action

        valids = self.kernel.check_state(state) # it is supposed to return a list now because

        action, m_idx = modify_mode(self.m, valids)
        # print(f"Valids: {valids} -> new action: {action}")
        self.safe_history.add_locations(init_action[0], action[0])
        
        if verbose:
            ex_kern_state = self.kernel.get_kernel_state(next_states[m_idx])
            inds = self.kernel.get_indices(next_states[m_idx])
            print(f"Expected (a: q{m_idx}- {action}) s': {next_states[m_idx]} -> s' kernel: {ex_kern_state} -> s' indices: {inds}")

        return action

    def action2mode(self, init_action):
        id = self.m.get_mode_id(init_action[1], init_action[0])
        return self.m.qs[id], id

    def check_init_action(self, state, init_action):
        d, v = init_action
        b = 0.523
        g = 9.81
        l_d = 0.329
        if abs(d)> 0.06: 
            #  only check the friction limit if it might be a problem
            friction_v = np.sqrt(b*g*l_d/np.tan(abs(d))) *1.1
            if friction_v < v:
                print(f"Invalid action: check planner or expect bad resultsL {init_action} -> max_friction_v: {friction_v}")
                return False, state

        m = self.m.get_mode_id(v, d)
        next_state = update_complex_state(state, init_action, self.time_step)
        safe = self.kernel.check_state(next_state)
        if safe[m]:
            return True, next_state
        elif not safe[m]:
            return False, next_state
        else:
            raise ValueError(f"Invalid kernel return: {safe}")


def modify_mode(self: Modes, valid_window):
    """ 
    modifies the action for obstacle avoidance only, it doesn't check the dynamic limits here.

    Returns
        Mode (v, delta)
        Mode_idx
    """
    # max_v_idx = 
    #TODO: decrease the upper limit of the search according to the velocity

    assert valid_window.any() == 1, "No valid actions:check modify_mode method"

    for vm in range(self.nq_velocity-1, -1, -1):
        idx_search = int(self.nv_modes[vm] +(self.nv_level_modes[vm]-1)/2) # idx to start searching at.

        if valid_window[idx_search]:
            return self.qs[idx_search], idx_search

        if self.nv_level_modes[vm] == 1:
            # if there is only one option and it is invalid
            continue

        # at this point there are at least 3 steer options
        d_search_size = int((self.nv_level_modes[vm]-1)/2)

        for dind in range(d_search_size+1): # for d_ss=1 it should search, 0 and 1.
            p_d = int(idx_search+dind)
            if valid_window[p_d]:
                return self.qs[p_d], p_d
            n_d = int(idx_search-dind-1)
            if valid_window[n_d]:
                return self.qs[n_d], n_d
        
    print(f"Idx_searh: {idx_search} -> vm: {vm} -> d_search_size: {d_search_size} -> dind: {dind}")
    print(f"No action found, window: {valid_window} n:{n_d} - p:{p_d}")
    raise ValueError("modify_mode: unable to find valid action")


class ForestSquidKernel():
    def __init__(self, sim_conf, plotting=False):
        # self.kernel = None
        # self.side_kernel = np.load(f"{sim_conf.kernel_path}SideKernel_{sim_conf.kernel_mode}.npy")
        # self.obs_kernel = np.load(f"{sim_conf.kernel_path}ObsKernel_{sim_conf.kernel_mode}.npy")

        self.turtle = None
        self.side_turtle = np.load(f"{sim_conf.kernel_path}SideTurtle_{sim_conf.kernel_mode}.npy")
        # self.side_turtle = 0
        self.obs_turtle = np.load(f"{sim_conf.kernel_path}ObsTurtle_{sim_conf.kernel_mode}.npy")

        # self.side_fish_tab = np.load(f"{sim_conf.kernel_path}FishTab_{sim_conf.kernel_mode}.npy")
        self.obs_fish_tab = np.load(f"{sim_conf.kernel_path}ObsFishTab_{sim_conf.kernel_mode}.npy")

        img_size = int(sim_conf.obs_img_size * sim_conf.n_dx)
        obs_size = int(sim_conf.obs_size * sim_conf.n_dx)
        self.obs_offset = int((img_size - obs_size) / 2)

        self.resolution = sim_conf.n_dx
        self.plotting = plotting
        self.m = Modes(sim_conf)
        self.sim_conf = sim_conf

    def construct_kernel(self, track_size, obs_locations):
        self.turtle = construct_forest_kernel(track_size, obs_locations, self.resolution, self.side_turtle, self.obs_turtle, self.obs_offset)
        # zes = np.count_nonzero(self.turtle)
        # print(f"Kernel constructed: {self.turtle.size - zes}")
        # print(f"Turtle Constructed: {self.turtle.shape}")

    def check_state(self, state):
        i, j, k, m = self.get_indices(state)
        turtle_val = self.turtle[i, j, k, m]
        if turtle_val == -1:
            return np.ones(self.m.n_modes)
        if turtle_val == -2:
            return np.zeros(self.m.n_modes)
        else:
            return self.obs_fish_tab[int(turtle_val)]

    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.turtle.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.turtle.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.turtle.shape[2]-1)))
        theta_ind = min(max(0, theta_ind), self.turtle.shape[2]-1)
        mode = self.m.get_mode_id(state[3], state[4])

        return x_ind, y_ind, theta_ind, mode


# @njit(cache=True)
def construct_forest_kernel(track_size, obs_locations, resolution, side_kernel, obs_kernel, obs_offset):
    kernel = np.ones((track_size[0], track_size[1], obs_kernel.shape[2], obs_kernel.shape[3]))
    kernel *= -1 # all actions are ok
    #! I have remove the sides from the kernel, so the kernel is now only for the obstacles
    # length = int(track_size[1] / resolution)
    # for i in range(length):
    #     kernel[:, i*resolution:(i+1)*resolution] = side_kernel

    if obs_locations is None:
        return kernel

    for obs in obs_locations:
        i = int(round(obs[0] * resolution)) - obs_offset
        j = int(round(obs[1] * resolution)) - obs_offset * 2
        if i < 0:
            # print(f"{i}: from {obs_kernel.shape[0]-kernel.shape[0]-abs(i)-abs(i)} to {i+obs_kernel.shape[0]}")
            # kernel[0:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] = obs_kernel[abs(i):obs_kernel.shape[0], :]
            kernel[0:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] = obs_kernel[abs(i):min(abs(i)+kernel.shape[0], obs_kernel.shape[0]), :]
            continue

        if kernel.shape[0] - i <= (obs_kernel.shape[0]):
            kernel.shape[0] - i
            kernel[i:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] = obs_kernel[0:kernel.shape[0] - i, :]
            continue


        kernel[i:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] = obs_kernel
    
    return kernel




