import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SandboxSafety.Simulator.Dynamics import update_std_state, update_complex_state, update_complex_state_const
from SandboxSafety.KernelTests.GeneralTestTrain import load_conf

class BaseKernel:
    def __init__(self, track_img, sim_conf):
        self.track_img = track_img
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.kernel_time_step
        self.n_phi = sim_conf.n_phi
        self.phi_range = sim_conf.phi_range
        self.half_block = 1 / (2*self.n_dx)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = sim_conf.n_modes 
        self.sim_conf = sim_conf
        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = np.linspace(-self.max_steer, self.max_steer, self.n_modes)[:, None] * np.ones((5, 2)) 
        self.n_modes *= 3
        self.qs = np.vstack((self.qs, self.qs, self.qs))
        self.qs[0:5, 1] = 1
        self.qs[5:10, 1] = 2
        self.qs[10:, 1] = 3 

        self.o_map = np.copy(self.track_img)    
        self.fig, self.axs = plt.subplots(2, 2)
        self.mode_window = 10

    def save_kernel(self, name):
        np.save(f"{self.sim_conf.kernel_path}{name}.npy", self.kernel)
        print(f"Saved kernel to file: {name}")

class ViabilityGenerator(BaseKernel):
    def __init__(self, track_img, sim_conf):
        super().__init__(track_img, sim_conf)
        
        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        
        self.kernel[:, :, :] = self.track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))

        self.dynamics = build_viability_dynamics(self.phis, self.qs, self.t_step, self.sim_conf.n_dx, self.sim_conf.phi_range)
        self.dynamics = self.dynamics.astype(np.int)

    def view_kernel(self, phi, show=True, fig_n=1):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(fig_n)
        plt.clf()
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k = self.dynamics[phi_ind, m, -1]


            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)
        if show:
            plt.show()
    
    def view_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)
        mode = 2

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
        plt.savefig("SandboxSafety/Kernels/Kernel_build.svg")

        if show:
            plt.show()

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = viability_loop(self.kernel, self.dynamics, self.mode_window)

            self.view_build(False)

@njit(cache=True)
def build_viability_dynamics(phis, qs, time, resolution, phi_range):
    l_phis = len(phis)
    l_qs = len(qs)

    dynamics = np.zeros((l_phis, l_qs, 3))
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            state = np.array([0, 0, p, m[1], 0])
            action = np.array([m[0], m[1]])
            new_state = update_complex_state(state, action, time)
            dx, dy, phi = new_state[0], new_state[1], new_state[2]

            if phi > np.pi:
                phi = phi - 2*np.pi
            elif phi < -np.pi:
                phi = phi + 2*np.pi
            new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
            dynamics[i, j, 2] = min(max(0, new_k), len(phis)-1)
            
            dynamics[i, j, 0] = int(round(dx * resolution))                  
            dynamics[i, j, 1] = int(round(dy * resolution))                  
            

    return dynamics

@njit(cache=True)
def viability_loop(kernel, dynamics, mode_window):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                if kernel[i, j, k] == 1:
                    continue 
                kernel[i, j, k] = check_viable_state(i, j, k, dynamics, previous_kernel, mode_window)

    return kernel

@njit(cache=True)
def check_viable_state(i, j, k, dynamics, previous_kernel, mode_window):
    l_xs, l_ys, l_phis = previous_kernel.shape
    # for l in get_mode_list(m, mode_window, n_modes):
    n_modes = dynamics.shape[1]
    for l in range(n_modes):
        di, dj, new_k = dynamics[k, l, :]
        new_i = min(max(0, i + di), l_xs-1)  
        new_j = min(max(0, j + dj), l_ys-1)

        if not previous_kernel[new_i, new_j, new_k]:
            return False
    return True

@njit(cache=True)
def get_mode_list(m, mode_window, n_modes):
    lower_ind = max(0, m-mode_window)
    upper_ind = min(n_modes, m+mode_window)
    return range(lower_ind, upper_ind)


"""
    External functions
"""

def prepare_track_img(sim_conf):
    file_name = 'maps/' + sim_conf.map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())
    img_resolution = yaml_file['resolution']
    map_img_path = 'maps/' + yaml_file['image']

    resize = int(sim_conf.n_dx * img_resolution)

    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float64)
    if len(map_img.shape) == 3:
        map_img = map_img[:, :, 0]
    map_img[map_img <= 128.] = 1.
    map_img[map_img > 128.] = 0.

    img = Image.fromarray(map_img.T)
    img = img.resize((map_img.shape[0]*resize, map_img.shape[1]*resize))
    img = np.array(img)
    map_img2 = img.astype(np.float64)
    map_img2[map_img2 != 0.] = 1.

    return map_img2



def build_track_kernel(conf):
  
  

    img = prepare_track_img(conf) 
    # settrace(my_tracer)
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel(100)
    kernel.save_kernel(f"TrackKernel_{conf.track_kernel_path}_{conf.map_name}")
    kernel.view_build(True)




if __name__ == "__main__":
    conf = load_conf("track_kernel")
    build_track_kernel(conf)


