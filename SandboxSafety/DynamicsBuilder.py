
from SandboxSafety.Utils import load_conf

from SandboxSafety.Simulator.Dynamics import update_complex_state
from SandboxSafety.Modes import Modes

import numpy as np
from matplotlib import pyplot as plt

import numpy as np
from numba import njit


def build_dynamics_table(sim_conf):
    m = Modes(sim_conf)
    phis = np.linspace(-sim_conf.phi_range/2, sim_conf.phi_range/2, sim_conf.n_phi)

    if sim_conf.kernel_mode == "viab":
        dynamics = build_viability_dynamics(phis, m, sim_conf.kernel_time_step, sim_conf)
        np.save("viab_dyns.npy", dynamics)
    elif sim_conf.kernel_mode == 'disc':
        dynamics = build_disc_dynamics(phis, m, sim_conf.kernel_time_step, sim_conf)
        np.save("disc_dyns.npy", dynamics)
    else:
        raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")


# @njit(cache=True)
def build_viability_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range

    ns = 2

    dynamics = np.zeros((len(phis), len(m), len(m), ns, 4), dtype=np.int)
    invalid_counter = 0
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                new_state = update_complex_state(state, action, time/2)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_safe_mode_id(vel, steer)

                if new_q is None:
                    invalid_counter += 1
                    dynamics[i, j, k, :, :] = np.nan # denotes invalid transition
                    print(f"Invalid dyns: phi_ind: {i}, s_mode:{j}, action_mode:{k}")
                    continue
    

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 0, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 0, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 0, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 0, 3] = int(new_q)                  
                

                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 1, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 1, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 1, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 1, 3] = int(new_q)                  
                
    print(f"Invalid transitions: {invalid_counter}")

    return dynamics


# @njit(cache=True)
def build_disc_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size 
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    ns = 1
    invalid_counter = 0
    dynamics = np.zeros((len(phis), len(m), len(m), 9, 4), dtype=np.int)
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_safe_mode_id(vel, steer)
                # new_q = m.get_mode_id(vel, steer)

                if new_q is None:
                    invalid_counter += 1
                    dynamics[i, j, k, :, :] = np.nan # denotes invalid transition
                    print(f"Invalid dyns: phi_ind: {i}, s_mode:{j}, action_mode:{k}")
                    continue


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

                new_state = update_complex_state(state, action, time/2)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 8, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 8, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 8, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 8, 3] = int(new_q)   

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

    print(f"Invalid counter: {invalid_counter}")
    print(f"Dynamics Table has been built: {dynamics.shape}")

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
    #TODO: this could just be 4 blocks. There is no phi discretisation going on here. Maybe
    #! this isn't workign
    return temp_dynamics



def build_dynamics_table(sim_conf):
    m = Modes(sim_conf)
    phis = np.linspace(-sim_conf.phi_range/2, sim_conf.phi_range/2, sim_conf.n_phi)

    if sim_conf.kernel_mode == "viab":
        dynamics = build_viability_dynamics(phis, m, sim_conf.kernel_time_step, sim_conf)
    elif sim_conf.kernel_mode == 'disc':
        dynamics = build_disc_dynamics(phis, m, sim_conf.kernel_time_step, sim_conf)
    else:
        raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")


    np.save(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy", dynamics)



if __name__ == "__main__":
    conf = load_conf("forest_kernel")

    build_dynamics_table(conf)

