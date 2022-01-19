
import yaml   
from argparse import Namespace

from matplotlib import pyplot as plt
import numpy as np
import csv, time



# Training functions
def train_kernel_episodic(env, vehicle, sim_conf, show=False):
    print(f"Starting Episodic Training: {vehicle.planner.name}")
    start_time = time.time()
    state, done = env.reset(False), False

    for n in range(sim_conf.train_n):
        a, fake_done = vehicle.plan(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.planner.agent.train(2)
        
        if done or fake_done:
            vehicle.done_entry(s_prime, env.steps)
            if show:
                env.render(wait=False)
                vehicle.safe_history.plot_safe_history()

            state = env.reset(False)

    vehicle.planner.t_his.print_update(True)
    vehicle.planner.t_his.save_csv_data()
    vehicle.planner.agent.save(vehicle.planner.path)
    vehicle.save_intervention_list()

    train_time = time.time() - start_time
    print(f"Finished Episodic Training: {vehicle.planner.name} in {train_time} seconds")

    return train_time 

def train_kernel_continuous(env, vehicle, sim_conf, show=False):
    print(f"Starting Continuous Training: {vehicle.planner.name}")
    start_time = time.time()
    state, done = env.reset(False), False

    for n in range(sim_conf.train_n):
        a, fake_done = vehicle.plan(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.planner.agent.train(2)
        
        if done:
            vehicle.fake_done(env.steps)
            if show:
                env.render(wait=False)
                vehicle.safe_history.plot_safe_history()

            done = False
            state = env.fake_reset()

    vehicle.planner.t_his.print_update(True)
    vehicle.planner.t_his.save_csv_data()
    vehicle.planner.agent.save(vehicle.planner.path)
    vehicle.save_intervention_list()

    train_time = time.time() - start_time
    print(f"Finished Continuous Training: {vehicle.planner.name} in {train_time} seconds")

    return train_time 

def train_baseline_vehicle(env, vehicle, sim_conf, show=False):
    start_time = time.time()
    state, done = env.reset(False), False
    print(f"Starting Baseline Training: {vehicle.name}")
    crash_counter = 0

    for n in range(sim_conf.train_n):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.agent.train(2)
        
        if done:
            vehicle.done_entry(s_prime)
            if show:
                env.render(wait=False)
            if state['reward'] == -1:
                crash_counter += 1

            state = env.reset(False)

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()
    vehicle.agent.save(vehicle.path)

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter


# Test Functions
def evaluate_vehicle(env, vehicle, sim_conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 

    state = env.reset(False)
    done, score = False, 0.0

    state = env.reset(False)
    done, score = False, 0.0
    for i in range(sim_conf.test_n):
        while not done:
            a = vehicle.plan_act(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p
        if show:
            env.render(wait=False, name=vehicle.name)
            vehicle.safe_history.plot_safe_history()

        if r == -1:
            crashes += 1
            print(f"({i}) Crashed -> time: {env.steps} ")
        else:
            completes += 1
            print(f"({i}) Complete -> time: {env.steps}")
            lap_times.append(env.steps)
        state = env.reset(False)
        
        done = False

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['name'] = vehicle.name
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict

def test_kernel_vehicle(env, vehicle, show=False, laps=100, add_obs=False, wait=False):
    crashes = 0
    completes = 0
    lap_times = [] 

    state = env.reset(add_obs)
    # env.render(False)
    done, score = False, 0.0
    for i in range(laps):
        vehicle.kernel.construct_kernel(env.env_map.map_img.shape, env.env_map.obs_pts)
        while not done:
            a = vehicle.plan(state)
            s_p, r, done, _ = env.step_plan(a)
            # print(f"Actual: {s_p['state']}")
            state = s_p
            # env.render(False)
        if show:
            # env.history.show_history()
            # vehicle.history.save_nn_output()
            env.render(wait=wait, name=vehicle.planner.name)
            # plt.pause(1)
            # vehicle.safe_history.plot_safe_history()

        if r == -1:
            crashes += 1
            print(f"({i}) Crashed -> time: {env.steps} ")
            plt.show()
        else:
            completes += 1
            print(f"({i}) Complete -> time: {env.steps}")
            lap_times.append(env.steps)
        state = env.reset(add_obs)
        
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times Avg: {np.mean(lap_times)} --> Std: {np.std(lap_times)}")

    return completes



# Admin functions
def save_conf_dict(dictionary, save_name=None):
    if save_name is None:
        save_name  = dictionary["name"]
    path = dictionary["vehicle_path"] + dictionary["name"] + f"/{save_name}_record.yaml"
    with open(path, 'w') as file:
        yaml.dump(dictionary, file)

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



