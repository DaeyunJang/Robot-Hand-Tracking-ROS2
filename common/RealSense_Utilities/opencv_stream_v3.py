import cv2
import pyrealsense2 as rs
from RealSense_Utilities.realsense_api.realsense_api import RealSenseCamera
from RealSense_Utilities.realsense_api.realsense_api import find_realsense
from RealSense_Utilities.realsense_api.realsense_api import frame_to_np_array
from RealSense_Utilities.realsense_api.post_processing.option import OptionType

# TODO:
#   - Create a dictionary to hold multiple rosbag file paths that can be the
#     key while the value can be a camera instantiation.  This would be for
#     when we want to view multiple rosbags at the same time.
#   - Add the ability to change filter values using key presses while viewing
#     the opencv Stream
#   - Add the ability to save the current filter configuration to a text file
#     or json 


# Initialize the camera
cameras = []
realsense_device = find_realsense()

for i in realsense_device:
    cameras.append(RealSenseCamera(device=i, adv_mode_flag=True))

apply_filter = False
apply_align = True

try:
    while True:
        # Get the frameset and other data to be loaded into the class attributes
        for i in range(len(cameras)):
            cameras[i].get_data()

            if apply_filter:
                cameras[i].filter_depth_data(enable_decimation=False,
                                             enable_spatial=True,
                                             enable_temporal=True,
                                             enable_hole_filling=False)
                frameset = cameras[i].filtered_frameset
            else:
                frameset = cameras[i].frameset

            # filtering block
            # frameset = cameras[i].decimation_filter.process(frameset).as_frameset()
            frameset = cameras[i].depth_to_disparity.process(frameset).as_frameset()
            frameset = cameras[i].spatial_filter.process(frameset).as_frameset()
            frameset = cameras[i].temporal_filter.process(frameset).as_frameset()
            frameset = cameras[i].disparity_to_depth.process(frameset).as_frameset()
            # frameset = cameras[i].hole_filling_filter.process(frameset).as_frameset()

            if apply_align:
                cameras[i].get_aligned_frames(frameset, aligned_to_color=True)
                depth_frame = cameras[i].depth_frame_aligned
                color_frame = cameras[i].color_frame_aligned
            else:
                depth_frame = frameset.get_depth_frame()
                color_frame = frameset.get_color_frame()

            cameras[i].color_image = frame_to_np_array(color_frame)
            cameras[i].colored_depthmap = frame_to_np_array(depth_frame, colorize_depth=True)
            cameras[i].depth_image = frame_to_np_array(depth_frame)

        # Show image
        for i in cameras:
            image_name = '{}_filtered depth'.format(i.device.get_info(rs.camera_info.serial_number))
            cv2.imshow(image_name, i.colored_depthmap)

        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        # if key & 0xFF == ord('d'):
        #     decimation_magnitude = cameras[0].decimation.options[OptionType.MAGNITUDE]
        #     cameras[0].decimation.increment(decimation_magnitude)
finally:
    for i in cameras:
        i.stop()
