from numba import njit
import numpy as np
from matplotlib import pyplot as plt



#Dynamics functions
@njit(cache=True)
def update_kinematic_state(x, u, dt, whlb=0.33, max_steer=0.4, max_v=7):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    dx = np.array([x[3]*np.sin(x[2]), # x
                x[3]*np.cos(x[2]), # y
                x[3]/whlb * np.tan(x[4]), # theta
                u[1], # velocity
                u[0]]) # steering

    new_state = x + dx * dt 

    # check limits
    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state


@njit(cache=True)
def control_system(state, action, max_v=7, max_steer=0.4, max_a=6.5, max_d_dot=3.2):
    """
    Generates acceleration and steering velocity commands to follow a reference
    Note: the controller gains are hand tuned in the fcn

    Args:
        v_ref: the reference velocity to be followed
        d_ref: reference steering to be followed

    Returns:
        a: acceleration
        d_dot: the change in delta = steering velocity
    """
    # clip action
    v_ref = min(action[1], max_v)
    d_ref = max(action[0], -max_steer)
    d_ref = min(action[0], max_steer)

    kp_a = 10
    a = (v_ref-state[3])*kp_a
    
    kp_delta = 40
    d_dot = (d_ref-state[4])*kp_delta

    # clip actions
    #TODO: temporary removal of dynamic constraints
    a = min(a, max_a)
    a = max(a, -max_a)
    d_dot = min(d_dot, max_d_dot)
    d_dot = max(d_dot, -max_d_dot)
    
    u = np.array([d_dot, a])

    return u

@njit(cache=True)
def update_complex_state(state, action, dt, plan_steps=10, whlb=0.33, max_steer=0.4, max_v=7):
    n_dt = dt/(plan_steps-1)

    for i in range(plan_steps):
        u = control_system(state, action)
        state = update_kinematic_state(state, u, n_dt, whlb, max_steer, max_v)
        # print(f"CMPLX:: Action: {u} --. New state: {state}")

    return state

@njit(cache=True)
def update_complex_state_const(state, action, dt, whlb=0.33, max_steer=0.4, max_v=7):
    n_dt = 0.01
    plan_steps = dt/n_dt

    for i in range(plan_steps):
        u = control_system(state, action)
        state = update_kinematic_state(state, u, n_dt, whlb, max_steer, max_v)
        # print(f"CMPLX:: Action: {u} --. New state: {state}")

    return state

def update_std_state(x, u, dt):
    n_steps = 10
    n_dt = dt/(n_steps-1)
    for i in range(n_steps):
        x = update_single_state(x, u, n_dt)

    return x

@njit(cache=True)
def update_single_state(x, u, dt, whlb=0.33, max_steer=0.4, max_v=7):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    # dv = (x[3] - u[1])/dt
    #note: the u velocity is being used in the update and the steering action, not the current value. 
    theta_update = x[2] +  ((u[1] / whlb) * np.tan(u[0]) * dt)
    dx = np.array([u[1] * np.sin(theta_update),
                u[1]*np.cos(theta_update),
                u[1] / whlb * np.tan(u[0]),
                0,
                0])

    new_state = x + dx * dt 

    # new_state[4] = min(new_state[4], max_steer)
    # new_state[4] = max(new_state[4], -max_steer)
    # new_state[3] = min(new_state[3], max_v)

    return new_state

"""
    Just for testing. not needed or used anywhere else
"""
def simple_updates(x0, u0, t_total, n_steps):
    x = np.copy(x0)
    x_list = [x]
    t_update = np.array(t_total/(n_steps-1))
    for i in range(n_steps):
        x = update_single_state(x, u0, t_update)
        x_list.append(x)
        print(f"Simple: Action: {u0} --. New state: {x}")


    return np.array(x_list)

def complex_updates(x0, u0, t_total, n_steps):

    x = np.copy(x0)
    x_list = [x]
    t_update = np.array(t_total/(n_steps-1))
    for i in range(n_steps):
        u = control_system(x, u0)
        x = update_kinematic_state(x, u, t_update)
        x_list.append(x)


    return np.array(x_list)

def compare_simple_complex():
    t = 0.2

    x0 = np.array([0, 0, 0, 2, 0])
    u0 = np.array([0.4, 2])
    
    x2s = complex_updates(x0, u0, t, 10)
    x1s = simple_updates(x0, u0, t, 10)

    plt.figure(1)
    plt.title('Dynamics Comparison')
    plt.plot(x1s[:,0], x1s[:,1])
    plt.plot(x2s[:,0], x2s[:,1])
    plt.legend(['simple', 'complex'])
    # plt.show()

def compare_simple_complex2():
    t = 0.2

    x0 = np.array([0, 0, 0, 2, 0])
    u0 = np.array([0.4, 2])
    
    x1s = update_std_state(x0, u0, t)
    x2s = update_complex_state(x0, u0, t, 10)

    plt.figure(1)
    plt.title('Dynamics Comparison')
    plt.plot(x1s[0], x1s[1], 'x', markersize=20)
    plt.plot(x2s[0], x2s[1], 'x', markersize=20)
    plt.legend(['simple', 'complex'])
    plt.show()

if __name__ == "__main__":

    compare_simple_complex()
    compare_simple_complex2()
