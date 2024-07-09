import queue

import numpy as np
import sys
import os
import time
import cv2
import multiprocessing as mp
from quick_queue import QQueue
import signal

sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))
from mediapipe_api.handlandmarks import handlandmarks
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array


def handlandmarks_with_realsense(queue_handpose, queue_points, queue_handpose_sub, queue_points_sub):
    hld = handlandmarks.MediaPipeHandLandmarkDetector(replace_threshold=50)

    frame_height, frame_width, channels = (480, 640, 3)  # (이미지 픽셀 수,픽셀수  , 채널 수)
    cameras = {} # 빈 딕셔너리 생성 파이썬에서 키-쌍을 저장하는 자료구조
    realsense_device = find_realsense()
    # If using several cameras, detecting camera's individual serial.
    for serial, devices in realsense_device:  #반복문 realsenss_device에서 시리얼 번호와 장치 정보를 가져와 초기화 후 재설정 , cameras 딕셔너리에 저장
        cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d455',
                                          color_stream_fps=30, depth_stream_fps=30,
                                          color_stream_height=frame_height, color_stream_width=frame_width,
                                          depth_stream_height=frame_height, depth_stream_width=frame_width,
                                          disable_color_auto_exposure=False)
        time.sleep(1) # 카메라 초기화 후 대기

    _, rs_main_camera = cameras.popitem() # 마지막 키 값을 제거 후 반환 rs_main에 저장 , 하나의 realsense 카메라를  사용하기 위함
    print(f"{rs_main_camera}")

    count = 0
    s_t = time.time()
    try:
        while True:
            try:
                rs_main_camera.get_data()
                # captured every frameset includes RGB frame and depth frame.
                frameset = rs_main_camera.frameset

                # required to align depth frame to RGB frame.
                rs_main_camera.get_aligned_frames(frameset, aligned_to_color=True)

                # applying filters in depth frame.
                frameset = rs_main_camera.depth_to_disparity.process(rs_main_camera.frameset)
                frameset = rs_main_camera.spatial_filter.process(frameset)
                frameset = rs_main_camera.temporal_filter.process(frameset)
                frameset = rs_main_camera.disparity_to_depth.process(frameset)
                frameset = rs_main_camera.hole_filling_filter.process(frameset).as_frameset()

                # It is recommended to use a copy of the RGB image frame.
                img_color = np.copy(frame_to_np_array(frameset.get_color_frame()))
                img_depth = np.copy(frame_to_np_array(frameset.get_depth_frame()))

                hld.hand_detection(img_color, img_depth, rs_main_camera.depth_intrinsics)

                empty = queue_handpose.empty()
                if queue_handpose.empty():
                    queue_handpose.put(hld.hand_pose)
                    queue_points.put(hld.canonical_points)
                empty = queue_handpose_sub.empty()
                if queue_handpose_sub.empty():
                    queue_handpose_sub.put(hld.hand_pose)
                    queue_points_sub.put(hld.canonical_points)

                # queue_handpose.put(hld.hand_pose)
                # queue_points.put(hld.canonical_points)
                # queue_handpose_sub.put(hld.hand_pose)
                # queue_points_sub.put(hld.canonical_points)

                count += 1

                # use opencv to visualize results.
                if hld.drawing_image is not None:
                    cv2.imshow('RealSense_front', hld.drawing_image)

                key = cv2.pollKey()
                if key & 0xFF == ord('q') or key == 27:
                    break

                # if cv2.waitKey(1) == ord('q'):
                #     break

            except RuntimeError as runexpt:
                print(runexpt, " frame skipped")
                continue
    finally:
        cv2.destroyAllWindows()
        rs_main_camera.stop()
        print("main process closed")

def signal_handler(sig, frame):
    print('Signal received, terminating process...')
    if hand_process.is_alive():
        hand_process.terminate()
        hand_process.join()
    sys.exit(1)

if __name__ == '__main__':
    queue1 = mp.Manager().Queue(maxsize=1)
    queue2 = mp.Manager().Queue(maxsize=1)
    queue3 = mp.Manager().Queue(maxsize=1)
    queue4 = mp.Manager().Queue(maxsize=1)
    hand_process = mp.Process(target=handlandmarks_with_realsense, args=(queue1, queue2, queue3, queue4))
    hand_process.start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        hand_process.join()
    except KeyboardInterrupt:
        hand_process.terminate()
        hand_process.join()
        print(f'hand landmark with realsense Process terminated')


