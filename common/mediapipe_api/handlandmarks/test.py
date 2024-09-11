import cv2 #영상처리를 위한 라이브러리
import numpy as np  # 백터 및 행렬 연산 라이브러리
import pyrealsense2 as rs  # realsense 카메라를 파이선으로 제어 가능 모듈
import time #시간관련
import os  #운영체제 관련 모듈 operating system
import matplotlib.pyplot as plt



from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

DEVICE = "cuda"  # 그래픽카드 할당 사용

#realsense sdk사용하여 딥스 카메라 픽셀좌표에서 물리좌표로 변환하는 함수  x:가로 y:세로 d:깊이 intr: 카메라 내부 특성(렌즈 초점거리, 광학중심등)
def convert_depth_to_phys_coord(xp, yp, depth, intr):
    result = rs.rs2_deproject_pixel_to_point(intr, [int(xp), int(yp)], depth)

    return result[0], result[1], result[2]  # 반환된 xyz좌표

def main():
    # camera environment setup/카메라 초기화 및 설정
    frame_height, frame_width, channels = (480, 640, 3)  # (이미지 픽셀 수,픽셀수  , 채널 수)

    cameras = {} # 빈 딕셔너리 생성 파이썬에서 키-쌍을 저장하는 자료구조
    realsense_device = find_realsense()
    widths = []
    heights = []
    # If using several cameras, detecting camera's individual serial.
    for serial, devices in realsense_device:  #반복문 realsenss_device에서 시리얼 번호와 장치 정보를 가져와 초기화 후 재설정 , cameras 딕셔너리에 저장
        cameras[serial] = RealSenseCamera(device=devices, adv_mode_flag=True, device_type='d455',
                                          color_stream_fps=30, depth_stream_fps=30,
                                          color_stream_height=frame_height, color_stream_width=frame_width,
                                          depth_stream_height=frame_height, depth_stream_width=frame_width,
                                          disable_color_auto_exposure=False)
        time.sleep(5) # 카메라 초기화 후 5초 대기

    _, rs_main = cameras.popitem() # 마지막 키 값을 제거 후 반환 rs_main에 저장 , 하나의 realsense 카메라를  사용하기 위함

    if rs_main is None:
        print("can't initialize realsense cameras")

    # main streaming part
    while True: #while 루프문으 사용하여 실시간 프레임 캡쳐 및 처리
        try:
            # To get real-time frame, using get_data() function to capturing frame. 실시간 프레임 캡쳐
            rs_main.get_data()

            # captured every frameset includes RGB frame and depth frame.
            frameset = rs_main.frameset

            # required to align depth frame to RGB frame.
            rs_main.get_aligned_frames(frameset, aligned_to_color=True)

            # applying filters in depth frame.
            frameset = rs_main.depth_to_disparity.process(rs_main.frameset)
            frameset = rs_main.spatial_filter.process(frameset)
            frameset = rs_main.temporal_filter.process(frameset)
            frameset = rs_main.disparity_to_depth.process(frameset)
            frameset = rs_main.hole_filling_filter.process(frameset).as_frameset()

            # It is recommended to use a copy of the RGB image frame.
            img_rs0 = np.copy(frame_to_np_array(frameset.get_color_frame()))

            # Same to depth frame.
            img_depth = np.copy(frame_to_np_array(frameset.get_depth_frame()))
            img_raw = np.copy(img_rs0)

            results = model.predict(img_rs0)
            im_array = results[0].plot()

            # use opencv to visualize results.
            resized_image = cv2.resize(im_array, dsize=(0, 0), fx=1, fy=1, interpolation=cv2.INTER_AREA)
            cv2.namedWindow('RealSense_front', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RealSense_front', resized_image.shape[1], resized_image.shape[0])
            cv2.imshow('RealSense_front', resized_image)


            key = cv2.pollKey()

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

        except RuntimeError as runexpt:
            print(runexpt, " frame skipped")
            continue

    rs_main.stop()
    print("main process closed")


if __name__ == "__main__":
    main()