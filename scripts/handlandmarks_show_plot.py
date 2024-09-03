import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import multiprocessing as mp
import threading
import time
import sys, os

class RealTimePlot():
    def __init__(self, queue_handpose, queue_points, num_points=21):
        self.num_points = num_points
        self.queue_handpose = queue_handpose
        self.queue_points = queue_points
        self.handpose = None
        self.hand_points = np.zeros((21, 3))

        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(131, projection='3d')
        self.scatter = self.ax.scatter([], [], [])

        self.ax_cp = self.fig.add_subplot(132, projection='3d')
        self.scatter_world_points = self.ax_cp.scatter([], [], [])

        self.ax_wp = self.fig.add_subplot(133, projection='3d')
        self.scatter_canonical_points = self.ax_wp.scatter([], [], [])

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=1, interval=10)
        self.count = 0
        self.s_t = time.time()
        self.pps = 0

    def update_plot(self, frame):
        if self.count == 1: # start
            self.s_t = time.time()
        try:
            # if self.queue_handpose.full():
            self.handpose = self.queue_handpose.get()
            self.hand_points = self.queue_points.get()
            # print(f"[Process-3D plot] handpose : {self.handpose}")
        except Exception as e:
            return

        self.pps = self.count / (time.time() - self.s_t)
        self.count += 1

        if self.handpose['landmarks'] is None or self.hand_points is None:
            return None

        self.ax.cla()
        self.ax_cp.cla()
        self.ax_wp.cla()

        self.ax.set_title('normalized')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        # self.ax.set_xlim(-1, 1)
        # self.ax.set_ylim(-1, 1)
        # self.ax.set_zlim(-1, 1)

        self.ax_cp.set_title('canonical')
        self.ax_cp.set_xlabel('X Label')
        self.ax_cp.set_ylabel('Y Label')
        self.ax_cp.set_zlabel('Z Label')
        self.ax_cp.set_xlim(0, 640)
        self.ax_cp.set_ylim(0, 480)
        self.ax_cp.set_zlim(200, 700)

        self.ax_wp.set_title('world coordinate')
        self.ax_wp.set_xlabel('X Label')
        self.ax_wp.set_ylabel('Y Label')
        self.ax_wp.set_zlabel('Z Label')
        self.ax_wp.set_xlim(-300, 300)
        self.ax_wp.set_ylim(-300, 300)
        self.ax_wp.set_zlim(200, 700)

        colors = ['black', 'blue', 'green', 'orange', 'red', 'black']
        intervals = [4, 8, 12, 16, 20]

        l_p = self.handpose['landmarks']
        w_p = self.handpose['world_landmarks']
        c_p = self.hand_points

        self.scatter = self.ax.scatter(l_p[:, 0], l_p[:, 1], l_p[:, 2], color='black', s=5, alpha=1)
        self.scatter_canonical_points = self.ax_cp.scatter(c_p[:, 0], c_p[:, 1], c_p[:, 2], color='black', s=5, alpha=1)
        self.scatter_world_points = self.ax_wp.scatter(w_p[:, 0], w_p[:, 1], w_p[:, 2], color='black', s=5, alpha=1)

        for i in range(len(intervals)):
            start_idx = 0 if i == 0 else intervals[i - 1] + 1
            end_idx = intervals[i]
            self.ax.plot(l_p[start_idx:end_idx + 1, 0], l_p[start_idx:end_idx + 1, 1], l_p[start_idx:end_idx + 1, 2], color=colors[i])
            self.ax_cp.plot(c_p[start_idx:end_idx + 1, 0], c_p[start_idx:end_idx + 1, 1], c_p[start_idx:end_idx + 1, 2], color='blue')
            self.ax_wp.plot(w_p[start_idx:end_idx + 1, 0], w_p[start_idx:end_idx + 1, 1], w_p[start_idx:end_idx + 1, 2], color='red')

    def plot_show(self):
        plt.show()

def start_real_time_plot(queue_handpose, queue_points):
    real_time_plot = RealTimePlot(queue_handpose, queue_points)
    real_time_plot.plot_show()

if __name__ == '__main__':
    queue_handpose = mp.Queue()
    queue_points = mp.Queue()
    plot_process = mp.Process(target=start_real_time_plot, args=(queue_handpose, queue_points))
    plot_process.start()
    plot_process.join()
