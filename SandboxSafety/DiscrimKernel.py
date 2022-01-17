import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SandboxSafety.Simulator.Dynamics import update_complex_state

from SandboxSafety.ViabKernel import BaseKernel
from SandboxSafety.Modes import Modes


class DiscrimGenerator(BaseKernel):
    def __init__(self, track_img, sim_conf):
        super().__init__(track_img, sim_conf)
        
        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))

        self.kernel[:, :, :, :] = track_img[:, :, None, None] * np.ones((self.n_x, self.n_y, self.n_phi, self.n_modes))

        self.dynamics = build_discrim_dynamics(self.phis, self.m, self.t_step, self.sim_conf)

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = viability_loop(self.kernel, self.dynamics)
            self.view_speed_build(False)
            # self.view_kernel(0, False, z)

        return self.get_filled_kernel()
     
    def view_kernel(self, phi, show=True, n=0):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k = self.dynamics[phi_ind, m, -1]
            # print(f"KernelDyns: Mode: {m} -> i, j: {di},{dj}")

            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)

        if show:
            plt.show()

    def make_kernel_img(self, phi, show=True, n=0):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
        plt.imshow(img, origin='lower')

        plt.pause(0.0001)

        plt.savefig(f"SandboxSafety/Kernels/Obs_build_{n}.svg")
        plt.xticks([])
        plt.yticks([])
        
        if show:
            plt.show()

    def view_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        # plt.title(f"Building Kernel")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()

    def view_speed_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        phi_ind = int(len(self.phis)/2)
        # phi_ind = 0
        # quarter_phi = int(len(self.phis)/4)
        # phi_ind = 

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, 2].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel speed: {2}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, 6].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel speed: {3}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, 8].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel speed: {4}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, 9].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel speed: {6}")

        # plt.title(f"Building Kernel")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
    

    def make_picture(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        self.axs[0, 0].set(xticks=[])
        self.axs[0, 0].set(yticks=[])
        self.axs[1, 0].set(xticks=[])
        self.axs[1, 0].set(yticks=[])
        self.axs[0, 1].set(xticks=[])
        self.axs[0, 1].set(yticks=[])
        self.axs[1, 1].set(xticks=[])
        self.axs[1, 1].set(yticks=[])

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        
        plt.pause(0.0001)
        plt.pause(1)
        plt.savefig(f"{self.sim_conf.kernel_path}Kernel_build_{self.sim_conf.kernel_mode}.svg")

        if show:
            plt.show()


# @njit(cache=True)
def build_discrim_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size *0.5
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    ns = 1
    dynamics = np.zeros((len(phis), len(m), len(m), 8*ns, 4), dtype=np.int)
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi < -np.pi:
                    phi = phi + 2*np.pi

                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                temp_dynamics = generate_temp_dynamics(dx, dy, h, resolution)
                
                dynamics[i, j, k, 0:8, 0:2] = np.copy(temp_dynamics)
                dynamics[i, j, k, 0:8, 3] = int(new_q) # no q discretisation error


                # new_state = update_complex_state(state, action, time*3/4)
                # dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                # new_q = m.get_mode_id(vel, steer)

                # if phi > np.pi:
                #     phi = phi - 2*np.pi
                # elif phi < -np.pi:
                #     phi = phi + 2*np.pi

                # new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                # dynamics[i, j, k, 8:8+4, 2] = min(max(0, new_k_min), len(phis)-1)
                
                # new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                # dynamics[i, j, k, 12:16, 2] = min(max(0, new_k_max), len(phis)-1)

                # temp_dynamics = generate_temp_dynamics(dx, dy, h, resolution)
                
                # dynamics[i, j, k, 8:, 0:2] = np.copy(temp_dynamics)
                # dynamics[i, j, k, 8:, 3] = int(new_q) # no q discretisation error


    return dynamics


@njit(cache=True)
def generate_temp_dynamics(dx, dy, h, resolution):
    temp_dynamics = np.zeros((8, 2))

    for i in range(2):
        temp_dynamics[0 + i*4, 0] = int(round((dx -h) * resolution))
        temp_dynamics[0 + i*4, 1] = int(round((dy -h) * resolution))
        temp_dynamics[1 + i*4, 0] = int(round((dx -h) * resolution))
        temp_dynamics[1 + i*4, 1] = int(round((dy +h) * resolution))
        temp_dynamics[2 + i*4, 0] = int(round((dx +h) * resolution))
        temp_dynamics[2 + i*4, 1] = int(round((dy +h )* resolution))
        temp_dynamics[3 + i*4, 0] = int(round((dx +h) * resolution))
        temp_dynamics[3 + i*4, 1] = int(round((dy -h) * resolution))

    return temp_dynamics

@njit(cache=True)
def viability_loop(kernel, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis, l_qs = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                for q in range(l_qs):
                    if kernel[i, j, k, q] == 1:
                        continue 
                    kernel[i, j, k, q] = check_viable_state(i, j, k, q, dynamics, previous_kernel)

    return kernel

@njit(cache=True)
def check_viable_state(i, j, k, q, dynamics, previous_kernel):
    l_xs, l_ys, l_phis, n_modes = previous_kernel.shape
    for l in range(n_modes):
        safe = True
        for n in range(dynamics.shape[3]): # cycle through 8 block states
            di, dj, new_k, new_q = dynamics[k, q, l, n, :]
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

            if previous_kernel[new_i, new_j, new_k, new_q]:
                # if you hit a constraint, break
                safe = False # breached a limit.
                break
        if safe:
            return False

    return True




def construct_obs_track(conf):
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx* 1.2) 
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, obs_offset:obs_size+obs_offset] = 1 
    # kernel = ViabilityGenerator(img, conf)
    kernel = DiscrimGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.view_build(True)

    kernel.save_kernel(f"ObsKernelTrack_{conf.track_kernel_path}")


if __name__ == "__main__":
    conf = load_conf("track_kernel")
    # conf.map_name = "race_track"
    build_track_discrim(conf)
    # construct_obs_track(conf)

    # conf = load_conf("forest_kernel")
    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

