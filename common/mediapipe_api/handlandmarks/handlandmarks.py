import mediapipe as mp
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os
import sys
import json
import queue
queue = queue.Queue()

class HandLandmarks():
    def __init__(self):
        script_dir = os.path.dirname(__file__)  # 현재 스크립트의 디렉토리 경로
        file_path = os.path.join(script_dir, 'handlandmark_keypoints.json')
        self.keypoints = json.load(open(file_path))
        # self.keypoints = json.load(open("handlandmark_keypoints.json"))
        self.hand_pose = {}  # tuple
        self.hand_pose['index'] = None
        self.hand_pose['score'] = None
        self.hand_pose['label'] = None
        self.hand_pose['landmarks'] = np.zeros((21, 3))
        self.hand_pose['world_landmarks'] = np.zeros((21, 3))  # calculate from camera instrinsic(Realsense)
        self.hand_results = []
        pass

class MediaPipeHandLandmarkDetector(HandLandmarks):
    # pass
    def __init__(self, image_width=640, image_height=480, replace_threshold=50):
        super().__init__()
        self.camera_to_hand_vector = {'x': 0, 'y': 0, 'z': 0}
        self.hand_landmarks = np.zeros((21, 3))
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8,
                                         min_tracking_confidence=0.7)
        self.canonical_points = np.zeros((21, 3))
        self.world_points = np.zeros((21, 3))
        self.world_points_iqr = np.zeros((21, 3))
        self.replace_points = np.zeros((21, 3))
        self.replace_threshold = replace_threshold

        self.palm_vector = {'x', 'y', 'z', 'r', 'p', 'y'}

        self.depth_intrinsics = None

        self.image_width = image_width
        self.image_height = image_height

        self.color_image = np.zeros((image_height, image_width, 3))
        self.depth_image = np.zeros((image_height, image_width))
        self.drawing_image = np.zeros((image_height, image_width, 3))   # show mediapipe algorithm image
        self.drawing_depth_image  = np.zeros((image_height, image_width, 3))   # show mediapipe algorithm image

        self.hand_thickness = 10  # mm
        self.pps = 0

    def __del__(self):
        pass

    def hand_detection(self, color_frame, depth_frame, depth_intrinsic=None):
        ######################################################
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        self.depth_intrinsics = depth_intrinsic
        self.color_image = color_frame
        # self.color_image = cv2.flip(self.color_image, 1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.depth_image = depth_frame
        self.depth_image_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.4), cv2.COLORMAP_JET)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        self.color_image.flags.writeable = False
        results = self.hands.process(self.color_image)

        # Draw the hand annotations on the image.
        self.color_image.flags.writeable = True
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR)

        self.drawing_image = self.color_image
        self.drawing_depth_image = self.depth_image_colormap

        self.hand_results.clear()
        self.hand_pose = {key: None for key in self.hand_pose}

        s_time = time.time()
        if results.multi_handedness:
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                # dictionary clear

                classification = handedness.classification[0]
                self.hand_pose['index'] = classification.index
                self.hand_pose['score'] = classification.score
                self.hand_pose['label'] = classification.label

                self.hand_landmarks = np.zeros((21, 3))
                for idx, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy, cz = landmrk.x, landmrk.y, landmrk.z
                    self.hand_landmarks[idx, 0] = cx
                    self.hand_landmarks[idx, 1] = cy
                    self.hand_landmarks[idx, 2] = cz

                canonical_points, world_points, world_points_iqr, replace_points, depth_avg = (
                    self.compare_coordinate_canonical_with_world(self.hand_landmarks))

                self.canonical_points = canonical_points
                self.world_points = world_points
                self.world_points_iqr = world_points_iqr
                self.replace_points = replace_points

                self.hand_pose['landmarks'] = self.hand_landmarks
                if self.depth_intrinsics != None:
                    self.hand_pose['world_landmarks'] = self.get_hand_world_xyz()
                c, n_v = self.get_palm_pose(coord='replace')
                c2, n_v2 = self.get_palm_pose(coord='canonical')

                # finally
                self.hand_results.append(self.hand_pose)

                # drawing points on image
                drawing_image = self.color_image
                self.mp_drawing.draw_landmarks(
                    drawing_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )

                img_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.3),
                                                            cv2.COLORMAP_JET)
                drawing_depth_image = self.depth_image_colormap
                self.mp_drawing.draw_landmarks(
                    drawing_depth_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style()
                )
                drawing_image = cv2.putText(img=drawing_image,
                                            text=f'Depth(avg) = {depth_avg}',
                                            org=(50, 50),
                                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                                            fontScale=1,
                                            color=(0,255,0),
                                            thickness=2,
                                            lineType=cv2.LINE_AA)
                drawing_image = cv2.putText(img=drawing_image,
                                            text=f'index finger depth = {world_points[7,2]}',
                                            org=(50, 80),
                                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                                            fontScale=1,
                                            color=(0, 255, 0),
                                            thickness=2,
                                            lineType=cv2.LINE_AA)
                self.drawing_image = drawing_image
                self.drawing_depth_image = drawing_depth_image

        e_time = time.time()
        if e_time != s_time:
            self.pps = 1 / (e_time - s_time)


    def compare_coordinate_canonical_with_world(self, hand_landmarks):
        """
        Comparing world points(from camera depth frame) to canonical points(from mediapipe values)
        :param hand_landmarks: (21,3) numpy array
        :param replace_th: threshold for replacing point to canonical values
        :return: list of values of all coordinates
        """
        canonical_point = np.zeros((21, 3))
        hand_landmarks = np.asarray(hand_landmarks)
        w, h = self.image_width, self.image_height

        canonical_point[:, 0] = [min(int(x * w), w - 1) for x in hand_landmarks[:, 0]]
        canonical_point[:, 1] = [min(int(x * h), h - 1) for x in hand_landmarks[:, 1]]
        canonical_point[:, 2] = [int(x * w) for x in hand_landmarks[:, 2]]
        canonical_point = np.asarray(canonical_point, dtype=int)

        world_point = np.zeros((21, 3))
        world_point_iqr = np.zeros((21, 3))

        # depth = np.zeros((1, 21))
        depth = self.depth_image[canonical_point[:, 1], canonical_point[:, 0]]
        world_point[:, 0] = canonical_point[:, 0]
        world_point[:, 1] = canonical_point[:, 1]
        world_point[:, 2] = depth
        world_point_iqr[:, 0] = canonical_point[:, 0]
        world_point_iqr[:, 1] = canonical_point[:, 1]
        world_point_iqr[:, 2] = self.replace_outliers_iqr_as_mean(depth, q1_rate=25, q3_rate=75, alpha=0)

        # world_point = np.asarray(world_point, dtype=int)
        world_point = np.asarray(world_point)
        # world_point = np.asarray(world_point_iqr)

        depth_avg = np.mean(world_point[:, 2])
        canonical_point[:, 2] = canonical_point[:, 2] + depth_avg + self.hand_thickness

        replace_points = world_point
        xyz_distances = np.linalg.norm(world_point - canonical_point, axis=1)
        replace_points[xyz_distances >= self.replace_threshold] = canonical_point[xyz_distances >= self.replace_threshold]
        return canonical_point, world_point, world_point_iqr, replace_points, depth_avg
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
                landmarks = self.replace_points
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
        """
        Obtain real distances x, y, z of each points from the camera in the world coordinate.
        :unit: mm
        :return: x, y, z values between each point and camera
        """
        xyz_world = []
        for idx, xyz in enumerate(self.replace_points):
            coordinate_xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [xyz[0], xyz[1]], xyz[2])
            xyz_world.append(coordinate_xyz)

        xyz_world = np.asarray(xyz_world, dtype=np.float64)
        return xyz_world

    def get_palm_pose(self, coord='replace'):
        """
        Find some points of hand landmarks
        :param keypoint: handlandmark_keypoints.json
        :param coord: canonical, replace, landmarks(normalized)
        :return: x, y, z values
        """
        p1 = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='wrist', coord=coord)
        p2 = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='pinky_mcp', coord=coord)
        p3 = self.get_hand_keypoint_pixel_xyz(hand_index=0, keypoint='index_finger_mcp', coord=coord)

        v1 = p2 - p1
        v2 = p3 - p1
        center_point = (p1 + p2 + p3) / 3.
        normal_vector = np.cross(v1, v2)
        normal_unit_vector = normal_vector / np.linalg.norm(normal_vector, 2)

        return center_point, normal_unit_vector

    @staticmethod
    def get_top_n_values(self, arr, n=3):
        arr = np.array(arr)
        max_indices = np.argpartition(arr, -n)[-n:]  # 가장 큰 값의 인덱스를 찾습니다.
        max_values = np.partition(arr, -n)[-n:]  # 배열에서 가장 큰 값 3개를 찾습니다.
        return max_indices, max_values

    @staticmethod
    def get_bottom_n_values(self, arr, n=3):
        arr = np.array(arr)
        min_indices = np.argpartition(arr, n)[:n]  # 가장 작은 값의 인덱스를 찾습니다.
        min_values = np.partition(arr, n)[:n]  # 배열에서 가장 작은 값 3개를 찾습니다.
        return min_indices, min_values

def main():
    HLD = MediaPipeHandLandmarkDetector()
    print(f"Created class: {HLD}")
    test_image = cv2.imread("hsh.jpg")

    arbitrary_depth_image = np.random.randint(0, 601, (480, 640))
    HLD.hand_detection(test_image, arbitrary_depth_image,)
    # cv2.imshow('test_image', test_image)
    cv2.imshow('mp_image', HLD.drawing_image)

    test_image = cv2.imread("wja1.png")

    arbitrary_depth_image = np.random.randint(0, 601, (480, 640))
    HLD.hand_detection(test_image, arbitrary_depth_image, )
    # cv2.imshow('test_image', test_image)
    cv2.imshow('mp2_image', HLD.drawing_image)

    test_image = cv2.imread("yj.jpg")

    arbitrary_depth_image = np.random.randint(0, 601, (480, 640))
    HLD.hand_detection(test_image, arbitrary_depth_image, )
    # cv2.imshow('test_image', test_image)
    cv2.imshow('mp3_image', HLD.drawing_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    sys.exit()

if __name__ == '__main__':
    main()

