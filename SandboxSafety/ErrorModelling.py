import numpy as np
from SandboxSafety.Simulator.Dynamics import update_complex_state
from SandboxSafety.Utils import load_conf
from SandboxSafety.Modes import Modes

def model_errors(conf):
    m = Modes(conf)
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    # h = conf.discrim_block * block_size 
    phi_size = phi_range / (conf.n_phi -1)
    # ph = conf.discrim_phi * phi_size
    time = conf.kernel_time_step


    b_state = [0, 0, 0, 3, 0]
    mode_action = [0.4, 2]
    action_id = m.get_mode_id(mode_action[1], mode_action[0])
    
    new_state = update_complex_state(b_state, mode_action, time)
    dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
    new_q = m.get_safe_mode_id(vel, steer)

    options = [[]]

def model_xy_error(conf):
    m = Modes(conf)
    resolution = conf.n_dx
    b = 1 / (resolution)
    time = conf.kernel_time_step


    b_state = np.array([0, 0, 0, 3.0, 0])
    mode_action = np.array([0.4, 2])
    # action_id = m.get_mode_id(mode_action[1], mode_action[0])
    
    n_bstate = update_complex_state(b_state, mode_action, time)
    dx, dy, phi, vel, steer = n_bstate[0], n_bstate[1], n_bstate[2], n_bstate[3], n_bstate[4]
    new_q = m.get_safe_mode_id(vel, steer)

    options = np.array([[b,b,0, 0, 0]
               ,[b,-b,0, 0, 0]
               ,[-b,b,0, 0, 0]
               ,[-b,-b,0, 0, 0]])

    new_states = []
    for i in range(4):
        state = b_state + options[i]
        new_state = update_complex_state(state, mode_action, time)
        new_states.append(new_state)

    for i, state in enumerate(new_states):
        diff = state - n_bstate
        print(f"{i} --> State diff: {diff} ")

def model_phi_error(conf):
    m = Modes(conf)
    p = conf.phi_range / (conf.n_phi -1)
    print(f"p: {p}")
    time = conf.kernel_time_step


    b_state = np.array([0, 0, 0, 3.0, 0])
    mode_action = np.array([0.4, 2])
    # action_id = m.get_mode_id(mode_action[1], mode_action[0])
    
    n_bstate = update_complex_state(b_state, mode_action, time)
    dx, dy, phi, vel, steer = n_bstate[0], n_bstate[1], n_bstate[2], n_bstate[3], n_bstate[4]
    new_q = m.get_safe_mode_id(vel, steer)

    options = np.array([[0,0,p, 0, 0]
               ,[0,0,-p, 0, 0]])

    new_states = []
    for i, opt in enumerate(options):
        state = b_state + opt
        new_state = update_complex_state(state, mode_action, time)
        new_states.append(new_state)

    for i, state in enumerate(new_states):
        diff = state - n_bstate
        print(f"{i} --> State diff: {diff} ")

def model_mode_error(conf):
    m = Modes(conf)
    v = 0.2
    # s = 0.04
    s = 0
    time = conf.kernel_time_step


    b_state = np.array([0, 0, 0, 4, -0.2])
    mode_action = np.array([0.2, 2])
    # action_id = m.get_mode_id(mode_action[1], mode_action[0])
    
    n_bstate = update_complex_state(b_state, mode_action, time)
    print(f"n_bstate: {n_bstate}")
    print(f"--------------------------------")
    dx, dy, phi, vel, steer = n_bstate[0], n_bstate[1], n_bstate[2], n_bstate[3], n_bstate[4]
    bq = m.get_safe_mode_id(vel, steer)

    options = np.array([[0,0,0, v, s]
               ,[0,0,0, -v, s]
               ,[0,0,0, v, -s]
               ,[0,0,0, -v, -s]])

    new_states = []
    for i, opt in enumerate(options):
        state = b_state + opt
        new_state = update_complex_state(state, mode_action, time)
        new_states.append(new_state)

    for i, state in enumerate(new_states):
        diff = state - n_bstate
        print(f"{i} --> State diff: {diff} ")
        qid = m.get_mode_id(state[3], state[4])
        print(f"bq: {bq}, Qid: {qid} ")



if __name__ == "__main__":
    conf = load_conf("forest_kernel")
    # model_errors(conf)
    # model_xy_error(conf)
    # model_phi_error(conf)
    model_mode_error(conf)