import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import multiprocessing as mp
from quick_queue import QQueue
import threading
import time
import sys, os

class RealTimePlot():
    def __init__(self, queue_handpose, queue_points, num_points=21):
        self.num_points = num_points
        self.queue_handpose = queue_handpose
        self.queue_points = queue_points

        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(231, projection='3d')
        self.scatter = self.ax.scatter([], [], [])

        self.ax_cp = self.fig.add_subplot(232, projection='3d')
        self.scatter_world_points = self.ax_cp.scatter([], [], [])

        self.ax_wp = self.fig.add_subplot(233, projection='3d')
        self.scatter_canonical_points = self.ax_wp.scatter([], [], [])

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=1, interval=10)
        self.count = 0
        self.s_t = time.time()
        self.pps = 0

    def update_plot(self, frame):
        if self.count == 1: # start
            self.s_t = time.time()
        try:
            # queue_handpose = self.queue_handpose.get_nowait()
            # queue_points = self.queue_points.get_nowait()
            queue_handpose = self.queue_handpose.get()
            queue_points = self.queue_points.get()
        except (mp.queues.Empty):
            # print(f'{mp.queues.Empty} queue empty')
            return

        # print(f'{queue_handpose}')
        # print(f'=============== Queue.get() ============================')
        # print(f'count = {self.count}')
        # print(f'[3D Plot Node] {time.time()} : {self.handpose}')
        # print(f'[Realsense Node] {time.time()} : {queue_points}')
        print(f'[3D plot] pps = {self.count / (time.time() - self.s_t)}')
        # print(f'========================================================')
        self.pps = self.count / (time.time() - self.s_t)
        self.count += 1

        # self.fig.text(0.95, 0.95, f'FPS: {self.pps:.2f}', ha='right', va='top',
        #               fontsize=12)

        if queue_handpose['landmarks'] is None or queue_points is None:
            return

        self.ax.cla()
        self.ax_cp.cla()
        self.ax_wp.cla()

        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        # self.ax.set_xlim(-1, 1)
        # self.ax.set_ylim(-1, 1)
        # self.ax.set_zlim(-1, 1)

        self.ax_cp.set_xlabel('X Label')
        self.ax_cp.set_ylabel('Y Label')
        self.ax_cp.set_zlabel('Z Label')
        self.ax_cp.set_xlim(300, 500)
        self.ax_cp.set_ylim(100, 300)
        self.ax_cp.set_zlim(400, 700)

        self.ax_wp.set_xlabel('X Label')
        self.ax_wp.set_ylabel('Y Label')
        self.ax_wp.set_zlabel('Z Label')
        self.ax_wp.set_xlim(300, 500)
        self.ax_wp.set_ylim(100, 300)
        self.ax_wp.set_zlim(400, 700)

        colors = ['black', 'blue', 'green', 'orange', 'red', 'black']
        intervals = [4, 8, 12, 16, 20]

        l_p = queue_handpose['landmarks']
        w_p = queue_handpose['world_landmarks']
        c_p = queue_points
        self.scatter = self.ax.scatter(l_p[:, 0], l_p[:, 1], l_p[:, 2], color='black', s=5, alpha=1)
        self.scatter_canonical_points = self.ax_cp.scatter(c_p[:, 0], c_p[:, 1], c_p[:, 2], color='black', s=5, alpha=1)
        self.scatter_world_points = self.ax_wp.scatter(w_p[:, 0], w_p[:, 1], w_p[:, 2], color='black', s=5, alpha=1)

        for i in range(len(intervals)):
            start_idx = 0 if i == 0 else intervals[i - 1] + 1
            end_idx = intervals[i]
            self.ax.plot(l_p[start_idx:end_idx + 1, 0], l_p[start_idx:end_idx + 1, 1], l_p[start_idx:end_idx + 1, 2], color=colors[i])
            self.ax_cp.plot(c_p[start_idx:end_idx + 1, 0], c_p[start_idx:end_idx + 1, 1], c_p[start_idx:end_idx + 1, 2], color='blue')
            self.ax_wp.plot(w_p[start_idx:end_idx + 1, 0], w_p[start_idx:end_idx + 1, 1], w_p[start_idx:end_idx + 1, 2], color='red')

        return
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