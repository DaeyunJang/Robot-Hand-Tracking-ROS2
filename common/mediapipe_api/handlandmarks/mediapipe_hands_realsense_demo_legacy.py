import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pyrealsense2 as rs
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from functools import partial
import threading
from multiprocessing import Process
# matplotlib.use('Agg')
import json
import queue

queue = queue.Queue()
g_hand_data = np.zeros((21, 3))
g_canonical_points = np.zeros((21, 3))
g_world_points = np.zeros((21, 3))
g_world_points_iqr = np.zeros((21, 3))
g_replace_point = np.zeros((21, 3))
g_palm_center = np.zeros(3)
g_palm_normal_vector = np.zeros(3)
g_palm_center_cano = np.zeros(3)
g_palm_normal_vector_cano = np.zeros(3)

class RealTimePlot3D:
    def __init__(self, num_points=21):
        self.num_points = num_points
        self.data = np.random.randn(self.num_points, 3)
        self.fig = plt.figure()
        # self.fig2 = plt.figure()
        self.ax = self.fig.add_subplot(231, projection='3d')    # visualization of hand pose
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        self.ax.set_xlim(-0.1, 0.1)
        self.ax.set_ylim(-0.1, 0.1)
        self.ax.set_zlim(-0.1, 0.1)

        self.ax2 = self.fig.add_subplot(232, projection='3d')  # visualization of hand pose
        self.ax2.set_xlabel('X pixel')
        self.ax2.set_ylabel('Y pixel')
        self.ax2.set_zlabel('Z depth')

        self.ax3 = self.fig.add_subplot(233, projection='3d')    # visualization of hand pose
        self.ax3.set_xlabel('X pixel')
        self.ax3.set_ylabel('Y pixel')
        self.ax3.set_zlabel('Z depth')

        self.ax4 = self.fig.add_subplot(234, projection='3d')  # visualization of hand pose
        self.ax4.set_xlabel('X pixel')
        self.ax4.set_ylabel('Y pixel')
        self.ax4.set_zlabel('Z depth')

        self.ax5 = self.fig.add_subplot(235, projection='3d')  # visualization of hand pose
        self.ax5.set_xlabel('X pixel')
        self.ax5.set_ylabel('Y pixel')
        self.ax5.set_zlabel('Z depth')

        self.ax6 = self.fig.add_subplot(236, projection='3d')  # visualization of hand pose
        self.ax6.set_xlabel('X pixel')
        self.ax6.set_ylabel('Y pixel')
        self.ax6.set_zlabel('Z depth')

        # self.ani = None
        self.ani = FuncAnimation(self.fig, self.update_data, frames=1, interval=100)
        self.scatter = self.ax.scatter(g_hand_data[:, 0], g_hand_data[:, 1], g_hand_data[:, 2])
        self.scatter_canonical_points = self.ax.scatter(g_canonical_points[:, 0], g_canonical_points[:, 1], g_canonical_points[:, 2])
        self.scatter_world_points = self.ax.scatter(g_world_points[:, 0], g_world_points[:, 1], g_world_points[:, 2])
        self.scatter_world_points_iqr = self.ax.scatter(g_world_points_iqr[:, 0], g_world_points_iqr[:, 1], g_world_points_iqr[:, 2])
        self.scatter_replace_points = self.ax.scatter(g_replace_point[:, 0], g_replace_point[:, 1], g_replace_point[:, 2])
        self.update_thread = threading.Thread(target=self.update_data)
        self.update_thread.daemon = True

    def update_data(self, frames):

        global g_hand_data, g_canonical_points, g_world_points, g_world_points_iqr, g_replace_point
        global g_palm_center, g_palm_normal_vector, g_palm_center_cano, g_palm_normal_vector_cano
        colors = ['black', 'blue', 'green', 'orange', 'red', 'black']
        intervals = [4, 8, 12, 16, 20]

        while True:

            data = g_hand_data
            data_c_p = g_canonical_points
            data_w_p = g_world_points
            data_w_p_iqr = g_world_points_iqr
            data_r_p = g_replace_point
            # print(f'canonical: {data_c_p}')
            # print(f'world_iqr: {data_w_p_iqr}')
            # print(f'replace: {data_r_p}')

            # 현재 점 삭제
            self.scatter.remove()
            self.scatter_canonical_points.remove()
            self.scatter_world_points.remove()
            self.scatter_world_points_iqr.remove()
            self.scatter_replace_points.remove()
            self.ax.cla()
            self.ax2.cla()
            self.ax3.cla()
            self.ax4.cla()
            self.ax5.cla()
            self.ax6.cla()
            self.ax.set_xlabel('X Label')
            self.ax.set_ylabel('Y Label')
            self.ax.set_zlabel('Z Label')
            # self.ax.set_xlim(0, 500)
            # self.ax.set_ylim(0, 500)
            # self.ax.set_zlim(-100, 400)

            max_xyz_c = np.max(data_c_p, axis=0)
            min_xyz_c = np.min(data_c_p, axis=0)
            max_xyz_w = np.max(data_w_p, axis=0)
            min_xyz_w = np.min(data_w_p, axis=0)
            max_xyz_w_iqr = np.max(data_w_p_iqr, axis=0)
            min_xyz_w_iqr = np.min(data_w_p_iqr, axis=0)

            th = 50
            # self.ax2.set_xlim(min_xyz_c[0] - th, max_xyz_c[0] + th)
            # self.ax2.set_ylim(min_xyz_c[1] - th, max_xyz_c[1] + th)
            # self.ax2.set_zlim(min_xyz_c[2] - th, max_xyz_c[2] + th)
            # self.ax4.set_xlim(min_xyz_c[0] - th, max_xyz_c[0] + th)
            # self.ax4.set_ylim(min_xyz_c[1] - th, max_xyz_c[1] + th)
            # self.ax4.set_zlim(min_xyz_c[2] - th, max_xyz_c[2] + th)
            # self.ax5.set_xlim(min_xyz_w[0] - th, max_xyz_w[0] + th)
            # self.ax5.set_ylim(min_xyz_w[1] - th, max_xyz_w[1] + th)
            # self.ax5.set_zlim(min_xyz_w[2] - th, max_xyz_w[2] + th)
            # self.ax6.set_xlim(min_xyz_w_iqr[0] - th, max_xyz_w_iqr[0] + th)
            # self.ax6.set_ylim(min_xyz_w_iqr[1] - th, max_xyz_w_iqr[1] + th)
            # self.ax6.set_zlim(min_xyz_w_iqr[2] - th, max_xyz_w_iqr[2] + th)

            self.ax2.set_xlim(300, 500)
            self.ax2.set_ylim(100, 300)
            self.ax2.set_zlim(400, 700)
            self.ax3.set_xlim(300, 500)
            self.ax3.set_ylim(100, 300)
            self.ax3.set_zlim(400, 700)
            self.ax4.set_xlim(300, 500)
            self.ax4.set_ylim(100, 300)
            self.ax4.set_zlim(400, 700)
            self.ax5.set_xlim(300, 500)
            self.ax5.set_ylim(100, 300)
            self.ax5.set_zlim(400, 700)
            self.ax6.set_xlim(300, 500)
            self.ax6.set_ylim(100, 300)
            self.ax6.set_zlim(400, 700)


            # print(f'{data[8,0]} / {data[8,1]} / {data[8,2]}')
            self.scatter = self.ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='black', s=50, alpha=1)

            self.scatter_canonical_points = self.ax2.scatter(data_c_p[:, 0], data_c_p[:, 1], data_c_p[:, 2], color='red', s=10, alpha=1)
            self.scatter_world_points = self.ax2.scatter(data_w_p[:, 0], data_w_p[:, 1], data_w_p[:, 2], color='blue', s=10, alpha=1)
            self.scatter_world_points_iqr = self.ax2.scatter(data_w_p_iqr[:, 0], data_w_p_iqr[:, 1], data_w_p_iqr[:, 2], color='green', s=10, alpha=1)
            self.scatter_replace_points = self.ax2.scatter(data_r_p[:, 0], data_r_p[:, 1], data_r_p[:, 2], color='purple', s=10, alpha=1)

            self.scatter_replace_points = self.ax3.scatter(data_r_p[:, 0], data_r_p[:, 1], data_r_p[:, 2], color='purple', s=10, alpha=1)

            self.scatter_canonical_points = self.ax4.scatter(data_c_p[:, 0], data_c_p[:, 1], data_c_p[:, 2], color='red', s=10, alpha=1)

            self.scatter_world_points = self.ax5.scatter(data_w_p[:, 0], data_w_p[:, 1], data_w_p[:, 2], color='blue', s=10, alpha=1)

            self.scatter_world_points_iqr = self.ax6.scatter(data_w_p_iqr[:, 0], data_w_p_iqr[:, 1], data_w_p_iqr[:, 2], color='green', s=10, alpha=1)
            # self.scatter_replace_points = self.ax6.scatter(data_r_p[:, 0], data_r_p[:, 1], data_r_p[:, 2], color='purple', s=10, alpha=1)

            # print(f'canonical: {data_c_p}')
            # print(f'world_iqr: {data_w_p_iqr}')
            # print(f'replace: {data_r_p}')
            # self.scatter_2 = self.ax2.scatter()
            for i in range(len(intervals)):
                start_idx = 0 if i == 0 else intervals[i - 1] + 1
                end_idx = intervals[i]
                self.ax.plot(data[start_idx:end_idx + 1, 0], data[start_idx:end_idx + 1, 1], data[start_idx:end_idx + 1, 2], color=colors[i])
                self.ax2.plot(data_c_p[start_idx:end_idx + 1, 0], data_c_p[start_idx:end_idx + 1, 1], data_c_p[start_idx:end_idx + 1, 2], color='red')
                self.ax2.plot(data_w_p[start_idx:end_idx + 1, 0], data_w_p[start_idx:end_idx + 1, 1], data_w_p[start_idx:end_idx + 1, 2], color='blue')
                self.ax2.plot(data_w_p_iqr[start_idx:end_idx + 1, 0], data_w_p_iqr[start_idx:end_idx + 1, 1], data_w_p_iqr[start_idx:end_idx + 1, 2], color='green')
                self.ax2.plot(data_r_p[start_idx:end_idx + 1, 0], data_r_p[start_idx:end_idx + 1, 1], data_r_p[start_idx:end_idx + 1, 2], color='purple')

                self.ax3.plot(data_r_p[start_idx:end_idx + 1, 0], data_r_p[start_idx:end_idx + 1, 1], data_r_p[start_idx:end_idx + 1, 2], color='purple')
                # 벡터 그리기
                self.ax3.quiver(g_palm_center[0], g_palm_center[1], g_palm_center[2] , g_palm_normal_vector[0], g_palm_normal_vector[1], g_palm_normal_vector[2], color='black', length=150, normalize=False)

                self.ax4.plot(data_c_p[start_idx:end_idx + 1, 0], data_c_p[start_idx:end_idx + 1, 1], data_c_p[start_idx:end_idx + 1, 2], color='red')
                self.ax4.quiver(g_palm_center_cano[0], g_palm_center_cano[1], g_palm_center_cano[2] , g_palm_normal_vector_cano[0], g_palm_normal_vector_cano[1], g_palm_normal_vector_cano[2], color='black', length=150, normalize=False)

                self.ax5.plot(data_w_p[start_idx:end_idx + 1, 0], data_w_p[start_idx:end_idx + 1, 1], data_w_p[start_idx:end_idx + 1, 2], color='blue')

                self.ax6.plot(data_w_p_iqr[start_idx:end_idx + 1, 0], data_w_p_iqr[start_idx:end_idx + 1, 1], data_w_p_iqr[start_idx:end_idx + 1, 2], color='green')
                # self.ax6.plot(data_r_p[start_idx:end_idx + 1, 0], data_r_p[start_idx:end_idx + 1, 1], data_r_p[start_idx:end_idx + 1, 2], color='purple')

            return
            # return self.scatter

    def start(self):
        # 데이터 업데이트 쓰레드 시작
        self.update_thread.start()
        self.ani = FuncAnimation(self.fig, self.update_data, frames=100, interval=30)


class HandLandmarks():
    def __init__(self):
        self.keypoints = json.load(open("handlandmark_keypoints.json"))
        self.hand_landmarks = {}  # tuple
        self.hand_landmarks['index'] = None
        self.hand_landmarks['score'] = None
        self.hand_landmarks['label'] = None
        self.hand_landmarks['landmarks'] = np.zeros((21, 3))
        self.hand_landmarks['world_landmarks'] = np.zeros((21, 3))  # calculate from camera instrinsic(Realsense)
        # self.hand_landmarks['canonical_landmarks'] = None  # calculate from camera instrinsic(Realsense)
        # self.hand_landmarks['replace_landmarks'] = None  # calculate from camera instrinsic(Realsense)


        # if serveral hands is detected, hand_results will has objects of each hands.
        # e.g. [{hand_landmarks#1}, {hand_landmarks#2}, ...]
        self.hand_results = []

        pass


class MediaPipeHandLandmarkDetector(HandLandmarks):
    # pass
    def __init__(self):
        super().__init__()
        self.camera_to_hand_vector = {'x': 0, 'y': 0, 'z': 0}
        self.hand_data = np.zeros((21, 3))
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8,
                                         min_tracking_confidence=0.7)
        self.canonical_points = np.zeros((21, 3))
        self.world_points = np.zeros((21, 3))
        self.world_points_iqr = np.zeros((21, 3))
        self.replace_point = np.zeros((21, 3))
        self.palm_vector = {'x', 'y', 'z', 'r','p','y'}

        # smoothing (LPF-low pass filter)
        self.depth_intrinsics = None
        self.finger_depth_prev = 0
        self.finger_depth_curr = 0
        self.filter_sensitivity = 0.3

        self.image_width = 640
        self.image_height = 480

        self.color_image = None
        self.depth_image = None
        self.depth_image_prev = None
        self.depth_colormap = None
        self.blended_image = None
        self.start_flag = False

        self.hand_thickness = 10 # mm

        self.pps = 0

    def hand_detection(self, color_frame, depth_frame, depth_intrinsic=None):
        global g_hand_data, g_canonical_points, g_world_points, g_world_points_iqr, g_replace_point
        global g_palm_center, g_palm_normal_vector, g_palm_center_cano, g_palm_normal_vector_cano
        ######################################################
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        self.depth_intrinsics = depth_intrinsic
        self.color_image = color_frame
        self.color_image = cv2.flip(self.color_image, 1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.depth_image = depth_frame

        if self.start_flag == False:
            self.depth_image_filtered = self.depth_image
        else:
            self.depth_image_filtered = self.filter_sensitivity * self.depth_image + (
                    1 - self.filter_sensitivity) * self.depth_image_prev

        self.depth_image_filtered = cv2.flip(self.depth_image_filtered, 1)
        self.depth_image_prev = self.depth_image_filtered
        # depth_norm_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        self.color_image.flags.writeable = False
        results = self.hands.process(self.color_image)
        self.image_height, self.image_width, _ = self.color_image.shape
        depth_image_height, depth_image_width = self.depth_image_filtered.shape
        if self.image_height != depth_image_height or self.image_width != depth_image_width:
            print(f'[mediapipe_hands.py | Warning] It does not match a size (H x W) of color & depth frame')
        # Draw the hand annotations on the image.
        self.color_image.flags.writeable = True
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)
        ######################################################

        self.hand_results.clear()

        if results.multi_handedness:

            s_time = time.time()

            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                # dictionary clear
                self.hand_landmarks = {key: None for key in self.hand_landmarks}

                classification = handedness.classification[0]
                index = classification.index
                score = classification.score
                label = classification.label
                self.hand_landmarks['index'] = classification.index
                self.hand_landmarks['score'] = classification.score
                self.hand_landmarks['label'] = classification.label

                hand_data = np.zeros((21, 3))
                for idx, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy, cz = landmrk.x, landmrk.y, landmrk.z
                    hand_data[idx, 0] = cx
                    hand_data[idx, 1] = cy
                    hand_data[idx, 2] = cz
                    g_hand_data[idx, 0] = cx
                    g_hand_data[idx, 1] = cy
                    g_hand_data[idx, 2] = cz
                    pixel_x = int(cx * self.image_width)
                    pixel_y = int(cy * self.image_height)

                canonical_points, world_points, world_points_iqr, replace_point = self.compare_coordinate_canonical_with_world(hand_data)
                g_canonical_points = canonical_points
                g_world_points = world_points
                g_world_points_iqr = world_points_iqr
                g_replace_point = replace_point
                self.canonical_points = canonical_points
                self.world_points = world_points
                self.world_points_iqr = world_points_iqr
                self.replace_point = replace_point


                self.hand_landmarks['landmarks'] = hand_data
                # world_landmarks = self.conversion_hand_keypoint_pixel_to_point(hand_data, top_point_n=None)
                # self.hand_landmarks['world_landmarks'] = world_landmarks
                self.hand_landmarks['world_landmarks'] = self.get_hand_world_xyz()
                c, n_v = self.get_palm_pose(coord='replace')
                g_palm_center = c
                g_palm_normal_vector = n_v

                c2, n_v2 = self.get_palm_pose(coord='canonical')
                g_palm_center_cano = c2
                g_palm_normal_vector_cano = n_v2

                # finally
                self.hand_results.append(self.hand_landmarks)

                # drawing points on image
                self.mp_drawing.draw_landmarks(
                    self.color_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )

                # self.depth_colormap = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint16)
                # self.depth_colormap[:, :, 0] = self.depth_image_filtered
                # self.depth_colormap = cv2.cvtColor(self.depth_colormap, cv2.COLOR_RGB2BGR)

                self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image_filtered, alpha=0.3), cv2.COLORMAP_JET)
                # self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_colormap, alpha=0.3), cv2.COLORMAP_JET)

                # 이미지 블렌딩
                alpha = 0.3  # 컬러 이미지의 투명도
                beta = 1 - alpha  # 깊이 이미지의 투명도
                gamma = 0  # 추가적인 밝기 조절
                self.blended_image = cv2.addWeighted(self.color_image, alpha, self.depth_colormap, beta, gamma)


                # self.mp_drawing.draw_landmarks(
                #     self.depth_colormap,
                #     hand_landmarks,
                #     self.mp_hands.HAND_CONNECTIONS,
                #     # mp_drawing_styles.get_default_hand_landmarks_style(),
                #     # mp_drawing_styles.get_default_hand_connections_style()
                # )

            e_time = time.time()
            self.pps = 1 / (e_time - s_time)

    def compare_coordinate_canonical_with_world(self, hand_landmarks_array, replace_th=50):
        canonical_point = np.zeros((21, 3))
        hand_landmarks_array = np.asarray(hand_landmarks_array)
        w, h = self.image_width, self.image_height

        # canonical_point[:, 0] = min(int(hand_landmarks_array[:, 0] * w), w - 1)
        # canonical_point[:, 1] = min(int(hand_landmarks_array[:, 1] * h), h - 1)
        # canonical_point[:, 2] = int(hand_landmarks_array[:, 2] * w)
        canonical_point[:, 0] = [min(int(x * w), w - 1) for x in hand_landmarks_array[:, 0]]
        canonical_point[:, 1] = [min(int(x * h), h - 1) for x in hand_landmarks_array[:, 1]]
        canonical_point[:, 2] = [int(x * w) for x in hand_landmarks_array[:, 2]]
        canonical_point = np.asarray(canonical_point, dtype=int)

        world_point = np.zeros((21, 3))
        world_point_iqr = np.zeros((21, 3))

        # depth = np.zeros((1, 21))
        depth = self.depth_image_filtered[canonical_point[:, 1], canonical_point[:, 0]]
        world_point[:, 0] = canonical_point[:, 0]
        world_point[:, 1] = canonical_point[:, 1]
        world_point[:, 2] = depth
        world_point_iqr[:, 0] = canonical_point[:, 0]
        world_point_iqr[:, 1] = canonical_point[:, 1]
        world_point_iqr[:, 2] = self.replace_outliers_iqr_as_mean(depth, q1_rate=25, q3_rate=75, alpha=0)

        # world_point = np.asarray(world_point, dtype=int)
        world_point = np.asarray(world_point)

        # depth_avg = np.mean(world_point[:, 2])
        depth_avg = np.mean(world_point_iqr[:, 2])
        canonical_point[:, 2] = canonical_point[:, 2] + depth_avg + self.hand_thickness

        replace_point = world_point_iqr
        xyz_distances = np.linalg.norm(world_point_iqr-canonical_point, axis=1)
        # print(f'canonical_point: {canonical_point}')
        # print(f'world_point_iqr: {world_point_iqr}')
        # print(f'replace_point(before): {replace_point}')
        replace_point[xyz_distances >= replace_th] = canonical_point[xyz_distances >= replace_th]
        # print(f'distances(th=20 mm): {xyz_distances} / {xyz_distances >= replace_th}')
        # print(f'replace_point(after): {replace_point}')
        return canonical_point, world_point, world_point_iqr, replace_point
        pass

    @staticmethod
    def vector_to_rpy(x, y, z):
        # Yaw (ψ)
        yaw = np.arctan2(y, x)
        # Pitch (θ)
        pitch = np.arctan2(-z, np.sqrt(x ** 2 + y ** 2))
        # Roll (φ) is typically set to zero for a single vector
        roll = 0.0
        return roll, pitch, yaw

    @staticmethod
    def remove_outliers_iqr(data, q1_rate=25., q3_rate=75., alpha=1.5):
        """
        # IQR 방법으로 이상치 제거
        :return:
        """
        q1 = np.percentile(data, q1_rate)
        q3 = np.percentile(data, q3_rate)
        iqr = q3 - q1
        lower_bound = q1 - (alpha * iqr)
        upper_bound = q3 + (alpha * iqr)
        return data[(data >= lower_bound) & (data <= upper_bound)]

    @staticmethod
    def replace_outliers_iqr_as_mean(data, q1_rate=25., q3_rate=75., alpha=0.5):
        """
        # IQR 방법으로 이상치 제거하고 평균값으로 대체하는 함수
        :return:
        """
        q1 = np.percentile(data, q1_rate)
        q3 = np.percentile(data, q3_rate)
        iqr = q3 - q1
        lower_bound = q1 - (alpha * iqr)
        upper_bound = q3 + (alpha * iqr)

        # 이상치의 인덱스를 저장
        outliers_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

        # 이상치가 아닌 값들로 평균을 계산
        mean_value = np.mean(data[(data >= lower_bound) & (data <= upper_bound)])

        # 이상치를 평균값으로 대체
        data[outliers_indices] = mean_value

        # print(f'IQR: q1:{q1}, q3:{q3}, iqr:{iqr}, l_b:{lower_bound}, u_b:{upper_bound}, indices:{outliers_indices}, mean:{mean_value}')
        return data


    def get_hand_keypoint_pixel_xyz(self, hand_index=0, keypoint='index_finger_tip', coord='replace'):
        try:
            keypoint_index = self.keypoints['keypoints'][keypoint]

            if coord == 'canonical':
                landmarks = self.canonical_points
                x = landmarks[keypoint_index, 0]
                y = landmarks[keypoint_index, 1]
                z = landmarks[keypoint_index, 2]
                return np.array([x, y, z])

            elif coord == 'replace':
                landmarks = self.replace_point
                x = landmarks[keypoint_index, 0]
                y = landmarks[keypoint_index, 1]
                z = landmarks[keypoint_index, 2]
                return np.array([x, y, z])

            elif coord == 'normalized':
                landmarks = self.hand_results[hand_index]['landmarks']
                x = landmarks[keypoint_index, 0]
                y = landmarks[keypoint_index, 1]
                z = landmarks[keypoint_index, 2]
                return np.array([x, y, z])

        except Exception as e:
            print(f'[mediapipe_hands.py] Exception error: {e}')
        finally:
            pass

    def get_hand_world_xyz(self):
        xyz_world = []
        for idx, xyz in enumerate(self.replace_point):
            coordinate_xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [xyz[0], xyz[1]], xyz[2])
            xyz_world.append(coordinate_xyz)

        xyz_world = np.asarray(xyz_world, dtype=np.float64)
        return xyz_world

    def get_palm_pose(self, coord='replace'):
        p1 = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='wrist', coord=coord)
        p2 = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='pinky_mcp', coord=coord)
        p3 = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='index_finger_mcp', coord=coord)

        v1 = p2 - p1
        v2 = p3 - p1
        center_point = (p1 + p2 + p3) / 3.
        normal_vector = np.cross(v1, v2)
        normal_unit_vector = normal_vector / np.linalg.norm(normal_vector, 2)

        return center_point, normal_unit_vector

    def conversion_hand_keypoint_pixel_to_point(self, keypoints_array, top_point_n=None):
        """Algorithm for annotating pixel.
        -- Not Used --
        Select 3(or more, optional) points closest to the camera.
        And, calculate of scale of x,y and z (distance_pixel : distance_world)
        Finally, obtain all 21 x,y and z dimension values of keypoints on real world

        Landmarks
        There are 21 hand landmarks, each composed of x, y and z coordinates.
        The x and y coordinates are normalized to [0.0, 1.0] by the image width and height, respectively.
        The z coordinate represents the landmark depth, with the depth at the wrist being the origin.
        The smaller the value, the closer the landmark is to the camera.
        The magnitude of z uses roughly the same scale as x.
        source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results


        Author: DY
        Args:
            keypoints_array (ndarray): (21, 3) array

        Returns:
            ndarray: (21,3) world dx,dy,dz coordinates
        """

        try:
            keypoint_arr = np.reshape(keypoints_array, (21, 3))  # must be check

            if top_point_n == None:
                align_z_keypoint_world_arr = []
                h, w = self.depth_image_filtered.shape
                for idx, pixel_xyz in enumerate(keypoint_arr):
                    x = min(int(pixel_xyz[0] * w), w - 1)  # prevent the index overflow
                    y = min(int(pixel_xyz[1] * h), h - 1)  # prevent the index overflow
                    if x <= 0 or x >= w or y <= 0 or y >= h:
                        ''' out of index '''
                        return

                    z = self.depth_image_filtered[y, x]
                    xyz_world = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], z)
                    align_z_keypoint_world_arr.append(xyz_world)

                align_z_keypoint_world_arr = np.asarray(align_z_keypoint_world_arr, dtype=np.float64)
                return align_z_keypoint_world_arr

            elif isinstance(top_point_n, int):
                z_arr = keypoint_arr[:, 2]  # depth values
                min_z = np.min(z_arr)
                min_z_index = np.argmin(z_arr)
                ##########################################################
                ## select top N values and obtain scale of x,y and z
                # - start
                b_ids, b_values = self.get_bottom_n_values(z_arr, n=top_point_n)

                selected_keypoint_arr = keypoint_arr[b_ids]

                selected_keypoint_world_arr = []
                offset_min_z = 0
                h, w = self.depth_image_filtered.shape
                for idx, pixel_xyz in enumerate(selected_keypoint_arr):
                    x = min(int(pixel_xyz[0] * w), w - 1)  # prevent the index overflow
                    y = min(int(pixel_xyz[1] * h), h - 1)  # prevent the index overflow
                    if x <= 0 or x >= w or y <= 0 or y >= h:
                        ''' out of index '''
                        return

                    z = self.depth_image_filtered[y, x]
                    if idx == min_z_index:
                        offset_min_z = z
                    xyz_world = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], z)
                    selected_keypoint_world_arr.append(xyz_world)

                s_x, s_y, s_z = self.get_scales_xyz_coordinate_to_world(selected_keypoint_arr=selected_keypoint_arr,
                                                                        selected_keypoint_world_arr=selected_keypoint_world_arr,
                                                                        sz_type='sx')
                # obtan scale of x,y and z
                # - end
                ##########################################################

                # calculate world xyz of all keypoints
                # There is the keypoint which the nearest keypoint from camera, we make its z-value as 0.
                # cf) Originally, wrist point is 0 on mediapipe algorithm.
                '''
                TODO
                Apply scale of x,y and z  
                and offset from top distance !!! 2024.05.13
                '''
                align_z_keypoint_arr = keypoint_arr
                align_z_keypoint_arr[:, 2] = keypoint_arr[:, 2] - min_z
                align_z_keypoint_world_arr = []
                for idx, pixel_xyz in enumerate(align_z_keypoint_arr):
                    up_x = int(pixel_xyz[0] * w)
                    up_y = int(pixel_xyz[1] * h)
                    x = min(int(pixel_xyz[0] * w), w - 1)  # prevent the index overflow
                    y = min(int(pixel_xyz[1] * h), h - 1)  # prevent the index overflow
                    if x <= 0 or x >= w or y <= 0 or y >= h:
                        ''' out of index '''
                        return
                    # z = self.depth_image_filtered[y, x]
                    z = pixel_xyz[2] * s_z + offset_min_z

                    print(f'scale: {s_z} : pixel_xyz={pixel_xyz[2]} / z={z}')
                    xyz_world = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], z)
                    align_z_keypoint_world_arr.append(xyz_world)

                align_z_keypoint_world_arr = np.asarray(align_z_keypoint_world_arr, dtype=np.float64)
                return align_z_keypoint_world_arr

        except Exception as e:
            print(f'[mediapipe_hands.py] Exception error: {e}')
        finally:
            pass

    def get_top_n_values(self, arr, n=3):
        arr = np.array(arr)
        max_indices = np.argpartition(arr, -n)[-n:]  # 가장 큰 값의 인덱스를 찾습니다.
        max_values = np.partition(arr, -n)[-n:]  # 배열에서 가장 큰 값 3개를 찾습니다.
        return max_indices, max_values

    def get_bottom_n_values(self, arr, n=3):
        arr = np.array(arr)
        min_indices = np.argpartition(arr, n)[:n]  # 가장 작은 값의 인덱스를 찾습니다.
        min_values = np.partition(arr, n)[:n]  # 배열에서 가장 작은 값 3개를 찾습니다.
        return min_indices, min_values

    def get_scales_xyz_coordinate_to_world(self, selected_keypoint_arr, selected_keypoint_world_arr, sz_type='sx'):
        """Obtain scales of x,y and z
        -- Not Used --
        scale = abs(real_distance / point_distance)
        It is the average of values that have several points and are selected as a combination of two points.
        (nC2)

        Args:
            selected_keypoint_arr (_type_): _description_
            selected_keypoint_world_arr (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_arr = np.array(selected_keypoint_arr)
        w_arr = np.array(selected_keypoint_world_arr)

        scale_xyz = []
        # n = len(selected_keypoint_arr)
        for idx, _ in enumerate(selected_keypoint_arr):
            for i in range(idx + 1, len(selected_keypoint_arr)):
                scales = abs((w_arr[idx] - w_arr[i]) / (n_arr[idx] - n_arr[i]))
                scale_xyz.append(scales)
        scale_xyz = np.asarray(scale_xyz)  # convert to numpay array
        scale_x = np.mean(scale_xyz[:, 0])
        scale_y = np.mean(scale_xyz[:, 1])
        if (sz_type == 'sx'):
            scale_z = scale_x
        else:
            scale_z = np.mean(scale_xyz[:, 2])

        return scale_x, scale_y, scale_z

    # def get_hand_keypoint_normalized_xyz(self, hand_index=0, keypoint='index_finger_tip'):
    #     try:
    #         keypoint_index = self.keypoints['keypoints'][keypoint]
    #         landmarks = self.hand_results[hand_index]['landmarks']
    #         x = landmarks[keypoint_index,0]
    #         y = landmarks[keypoint_index,1]
    #         z = landmarks[keypoint_index,2]

    #         return x,y,z
    #     except Exception as e:
    #         print(f'[mediapipe_hands.py] Exception error: {e}')
    #     finally:
    #         pass

    def rs_init(self, fps=30):
        # Realsense 카메라 객체 생성
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        width = self.image_width
        height = self.image_height
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 카메라 시작
        self.profile = self.pipeline.start(self.config)
        # self.align = rs.align(rs.stream.color)

        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # color_intrinsics = align.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # depth_intrinsics = align.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        print(f'intrinsics(color): {self.color_intrinsics}')
        print(f'intrinsics(depth): {self.depth_intrinsics}')

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        if self.depth_sensor.supports(rs.option.depth_units):
            self.depth_sensor.set_option(rs.option.depth_units, 0.001)
            print(f'depth sensor: {self.depth_sensor} - change depth_units to : 0.001')
            self.pipeline.stop()
            print(f'Reopen the camera...')
            time.sleep(1)

            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
        else:
            print('depth sensor doesn''t support changing depth_units.')

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            raise RuntimeError("Could not acquire aligned frames.")

        '''
        TODO depth scale - D400: 0 L515: /4
        '''
        # self.depth_image = np.asanyarray(aligned_depth_frame.get_data()) / 4
        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        # self.color_image = np.asanyarray(color_frame.get_data())
        # self.depth_image = np.asanyarray(depth_frame.get_data())

        # depth_to_disparity_transform = rs.disparity_transform(True)  # True for depth to disparity
        # align_frames = depth_to_disparity_transform.process(align_frames)
        #
        # align_frames = rs.spatial_filter().process(align_frames)
        # align_frames = rs.temporal_filter().process(align_frames)
        #
        # disparity_to_depth_transform = rs.disparity_transform(False)
        # align_frames = disparity_to_depth_transform.process(align_frames).as_frameset()


        # return
        return self.color_image, self.depth_image

    def realsense_demo(self):
        global g_hand_data
        try:
            image_width = self.image_width
            image_height = self.image_height
            # mphand = MediaPipeHandLandmarkDetector()
            self.rs_init()

            while True:
                image_color, image_depth = self.get_frames()

                self.hand_detection(image_color, image_depth, self.depth_intrinsics)

                index_finger_tip_image = self.hand_landmarks['landmarks']
                index_finger_tip_world = self.hand_landmarks['world_landmarks']

                if index_finger_tip_image is None or index_finger_tip_world is None:
                    continue
                xyz_idx = index_finger_tip_image[8]
                x = min(int(xyz_idx[0] * image_width), image_width - 1)  # prevent the index overflow
                y = min(int(xyz_idx[1] * image_height), image_height - 1)  # prevent the index overflow
                # if x <= 0 or x >= w or y <= 0 or y >= h:
                #     ''' out of index '''
                #     return
                z = self.depth_image_filtered[y, x]
                z2 = index_finger_tip_world[8][2]

                finger_depth = z

                try:
                    m_f = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='middle_finger_tip', coord='canonical')
                    m_f_z = self.depth_image_filtered[m_f[1], m_f[0]]
                    cv2.putText(self.color_image, f"{m_f_z:.1f}", (m_f[0], m_f[1] - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    m_f = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='ring_finger_tip', coord='canonical')
                    m_f_z = self.depth_image_filtered[m_f[1], m_f[0]]
                    cv2.putText(self.color_image, f"{m_f_z:.1f}", (m_f[0], m_f[1] - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    m_f = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='pinky_tip', coord='canonical')
                    m_f_z = self.depth_image_filtered[m_f[1], m_f[0]]
                    cv2.putText(self.color_image, f"{m_f_z:.1f}", (m_f[0], m_f[1] - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                except Exception as e:
                    continue
                    pass


                cv2.line(self.color_image, (int(image_width / 2), int(image_height / 2)),
                         (int(image_width / 2), int(image_height / 2)), (255, 0, 0), 5)
                cv2.putText(self.color_image, f"{finger_depth:.1f} mm",
                            (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # cv2.putText(self.color_image, f"(scale: {z2:.1f}) mm",
                #             (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # cv2.putText(self.color_image,
                #             f"{x:.1f}, {y:.1f}, {z:.1f} mm",
                #             (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.line(self.depth_image_filtered, (x, y), (x, y), (0, 255, 0), 5)
                cv2.putText(self.depth_image_filtered, f"{finger_depth:.1f} mm",
                            (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                # cv2.putText(self.depth_image_filtered,
                #             f"{x:.1f}, {y:.1f}, {z:.1f} mm",
                #             (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)



                
                # if self.color_image is None or self.depth_colormap is None:
                #     continue

                cv2.imshow('MediaPipe Hands', self.color_image)

                # cv2.imshow('depth image', depth_show_img)
                
                cv2.imshow('depth image2', self.depth_colormap)
                cv2.imshow('depth image3', self.blended_image)



                if cv2.waitKey(5) & 0xFF == 27:
                    break
        except KeyboardInterrupt:
            print(f'Keyboard Interrupt (SIGINT)')
        finally:
            pass

def main():
    HLD = MediaPipeHandLandmarkDetector()
    thread_mediapipe_hand = threading.Thread(target=HLD.realsense_demo)
    thread_mediapipe_hand.daemon = True
    thread_mediapipe_hand.start()

    print('RealTimePlot3D class is update...')

    real_time_plot = RealTimePlot3D()
    real_time_plot.start()

    print('show start')
    plt.show()
    print('show end')

    # t = Process(target=HLD.realsense_demo(), args=(10,))
    # t.start()
    # t.join()

if __name__ == '__main__':
    main()

