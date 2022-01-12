import numpy as np
from numba import njit, jit
import csv 
from matplotlib import pyplot as plt
from auto_race_f110_gym.Utils import LibFunctions as lib
from auto_race_f110_gym.Utils import pure_pursuit_utils

class Trajectory:
    def __init__(self, map_name):
        self.map_name = map_name
        self.waypoints = None
        self.vs = None
        self.load_csv_track()
        self.n_wpts = len(self.waypoints)

        self.max_reacquire = 20

        self.diffs = None 
        self.l2s = None 
        self.ss = None 
        self.o_points = None

    def load_csv_track(self):
        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        # these get expanded
        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5]

        # these don't get expanded
        self.N = len(track)
        self.o_pts = np.copy(self.waypoints)
        self.ss = track[:, 0]
        self.diffs = self.o_pts[1:,:] - self.o_pts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 
        # self.show_trajectory()

        self._expand_wpts()


    def _expand_wpts(self):
        n = 5 # number of pts per orig pt 
        #TODO: make this a parameter
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

    def get_current_waypoint(self, position, lookahead_distance):
        #TODO: for compuational efficiency, pass the l2s and the diffs to the functions so that they don't have to be recalculated
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
            raise Exception("Waypoint not found")

    def show_trajectory(self):
        plt.figure(1)
        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1])
        plt.plot(self.waypoints[0, 0], self.waypoints[0, 1], 'x', markersize=20)
        plt.gca().set_aspect('equal', 'datalim')

        # plt.pause(0.0001)
        plt.show()

    def get_pose_progress(self, pose):
        return calculate_progress(pose, self.o_points, self.diffs, self.l2s, self.ss)


@njit(cache=True)
def calculate_progress(point, wpts, diffs, l2s, ss):
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
    s = s / ss[-1]

    return s 

