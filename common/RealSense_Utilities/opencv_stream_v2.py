from object_detection import ObjectDetect
import cv2
from realsense_api import RealSenseCamera

# TODO: 
#   - Restore the object detection functionality 
#   - Create a dictionary to hold multiple rosbag file paths that can be the 
#     key while the value can be a camera instantiation.  This would be for
#     when we want to view multiple rosbags at the same time.
#   - Add the ability to change filter values using key presses while viewing
#     the opencv Stream 
#   - Add the ability to save the current filter configuration to a text file 
#     or json 

#ros_bag = "C:\\Users\\35840\\Documents\\20211217_204044.bag"
# ros_bag = "C:\\Users\\35840\\Documents\\20211220_124541.bag"
ros_bag = "C:\\Users\\35840\\Documents\\20211220_132508.bag"
ros_bag2 = "C:\\Users\\35840\\Documents\\20211221_120843_2d.bag"
ros_bag3 = "C:\\Users\\35840\\Documents\\20211221_121055_3d.bag"

# Initialize the camera
# camera = RealSenseCamera(ros_bag2)
# camera2 = RealSenseCamera(ros_bag3)
camera = RealSenseCamera()


apply_filter = True

try:
    while True:
        camera.get_data() # Load the object's variables with data
        # camera2.get_data() # Load the object's variables with data
        
        # apply filtering to depth data
        if apply_filter:
            camera.filter_depth_data(enable_decimation=True,
                                    enable_spatial=True,
                                    enable_temporal=True,
                                    enable_hole_filling=True)
            filtered_frameset = camera.filtered_frameset

            # camera2.filter_depth_data(enable_decimation=True,
            #                         enable_spatial=True,
            #                         enable_temporal=True,
            #                         enable_hole_filling=False)
            # filtered_depth_frame2 = camera2.filtered_frameset.get_depth_frame()

        camera.get_aligned_frames(filtered_frameset, aligned_to_color=True)

        #depth_frame = camera.depth_frame
        filtered_depth_frame = camera.depth_frame_aligned

        color_frame = camera.color_frame_aligned
        # infrared_frame = camera.infrared_frame
        # color_intrin = camera.color_intrinsics

        # depth_frame = camera.depth_frame
        # color_frame = camera.color_frame
        # infrared_frame = camera.infrared_frame
        # color_intrin = camera.color_intrinsics

        # depth_frame2 = camera2.depth_frame
        # color_frame2 = camera2.color_frame
        # infrared_frame2 = camera2.infrared_frame
        # color_intrin2 = camera2.color_intrinsics

        color_img = frame_to_np_array(color_frame)
        colored_depth_image = frame_to_np_array(filtered_depth_frame, colorize_depth=True)
        depth_image = frame_to_np_array(filtered_depth_frame)

        #depth_image = camera.frame_to_np_array(depth_frame, colorize_depth=True)

        detector = ObjectDetect(color_img, depth_image, camera.depth_scale)
        detector.detect()
        detector.draw_rectangle(color_img)

        image_name = 'filtered depth'
        # proc_depth_image2 = camera2.frame_to_np_array(filtered_depth_frame2, colorize_depth=True)

        # depth_image2 = camera2.frame_to_np_array(depth_frame2, colorize_depth=True)

        # img2 = proc_depth_image2
        # image_name2 = 'filtered depth2'
        #img = cv2.resize(image_to_be_shown, (640, 480))
        #depth_image = cv2.resize(depth_image, (640, 480))

        cv2.imshow(image_name, colored_depth_image)
        #cv2.imshow('o', depth_image)
        # cv2.imshow(image_name2, img2)
        cv2.imshow('o2', color_img)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    camera.stop()