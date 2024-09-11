import cv2
from object_detection import ObjectDetect
import math
import numpy as np

from realsense_api import RealSenseCamera

top = 100
bottom = 100
left = 100
right = 100

BLUE = [255,0,0]

ros_bag = "C:\\Users\\35840\\Documents\\20211217_204044.bag"

def measure_dimensions(points):

    height = {'ix': points[0],
              'iy': points[2],
              'x' : points[0],
              'y' : points[3]}

    width = {'ix': points[0],
             'iy': points[2],
             'x' : points[1],
             'y' : points[2]}

    #dimensions = [height, width]

    #print(points)
    global color_intrin

    # Height
    udist_height = camera.depth_frame.get_distance(height['ix'],height['iy'])
    vdist_height = camera.depth_frame.get_distance(height['x'], height['y'])
    #print(udist,vdist)

    point1_height = rs.rs2_deproject_pixel_to_point(color_intrin, [height['ix'],height['iy']], udist_height)
    point2_height = rs.rs2_deproject_pixel_to_point(color_intrin, [height['x'], height['y']], vdist_height)
    #print(str(point1)+ ' ' +str(point2))

    height = math.sqrt(
                math.pow(point1_height[0] - point2_height[0], 2) + math.pow(point1_height[1] - point2_height[1],2) +
                math.pow(point1_height[2] - point2_height[2], 2)
                )

    # Width
    udist_width = camera.depth_frame.get_distance(width['ix'],width['iy'])
    vdist_width = camera.depth_frame.get_distance(width['x'], width['y'])
    #print(udist,vdist)

    point1_width = rs.rs2_deproject_pixel_to_point(color_intrin, [width['ix'],width['iy']], udist_width)
    point2_width = rs.rs2_deproject_pixel_to_point(color_intrin, [width['x'], width['y']], vdist_width)
    #print(str(point1)+ ' ' +str(point2))

    width = math.sqrt(
                math.pow(point1_width[0] - point2_width[0], 2) + math.pow(point1_width[1] - point2_width[1],2) +
                math.pow(point1_width[2] - point2_width[2], 2)
                )
    
    return height, width

# Initialize the camera
camera = RealSenseCamera(ros_bag)

display_dimensions = False
show_filtered_distance = False
show_filtered = False
apply_filter = True

try:
    while True:
        # try:
        camera.get_data() # Load the object's variables with data
        depth_frame = camera.depth_frame
        color_frame = camera.color_frame
        infrared_frame = camera.infrared_frame
        color_intrin = camera.color_intrinsics

        # apply filtering to depth data
        if apply_filter:
            camera.filter_depth_data(enable_decimation = True,
                                    enable_spatial = False,
                                    enable_temporal = True,
                                    enable_hole_filling = False)

            depth_frame = camera.processed_depth_frame
            print('filters applied')
            #processed_depth_image = camera.frame_to_np_array(processed_depth_frame, colorize_depth=True)

        depth_image = frame_to_np_array(depth_frame, colorize_depth=True)
        color_image = frame_to_np_array(color_frame)
        infrared_image = frame_to_np_array(infrared_frame)

        ###########################################
        ##---------- IMAGE TO BE SHOWN ----------##
        ###########################################
        image_to_be_shown = depth_image
        image_name = 'filtered depth'
        
        # cv2.imshow('color 640x480', color_image)
        # cv2.imshow('depth 320x240', depth_image)
        # cv2.imshow('IR 320x240', infrared_image)
        a = cv2.resize(image_to_be_shown, (640, 480))
        cv2.imshow(image_name, a)
        

        # size = cv2.getWindowImageRect('color 640x480')
        # print(size)

        # ret, depth_frame, color_frame, colorized_depth, color_intrin, filtered_depth_colored, filtered_depth = camera.get_frame()  
        # nn = ObjectDetect(color_frame, colorized_depth, depth_frame, camera.depth_scale, filtered_depth_colored, filtered_depth)
        #crop_img, depth_img, text, text_filt, text_location, points, confidence = nn.detect()

        #cv2.namedWindow('Both Streams', cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty('Both Strams', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #bordered = cv2.copyMakeBorder(stretched, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLUE)

        # if show_filtered_distance:
        #     cv2.putText(color_frame, text_filt, text_location,
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
        # else:
        #     print(confidence)
        #     cv2.putText(color_frame, text, text_location,
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
        #     cv2.putText(colorized_depth, text, text_location,
        #         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

        # if display_dimensions:
        #     height, width = measure_dimensions(points)
        #     print(height, width)

        #     if height != 0.0 or width != 0.0:
        #         width = f'width = {str(round(width, 3))} meters'
        #         height = f'height = {str(round(height, 3))} meters' 
        #         text_height_location = (points[1], (int((points[3] + points[2]) / 2)))
        #         text_width_location = (points[0], points[3] + 20)
        #         cv2.putText(color_frame, height, text_height_location,
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
        #         cv2.putText(color_frame, width, text_width_location,
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

        #images = np.hstack((color_frame, colorized_depth))
        #images_filtered = np.hstack((color_frame, filtered_depth_colored))
        #stretched = cv2.resize(images, (2560, 1440))

        #cv2.imshow("Filtered", images_filtered)
        
        if show_filtered:
            cv2.imshow('Filtered', filtered_depth_colored)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('m'): #enable dimensions measurements
            display_dimensions = True
        elif key & 0xFF == ord('o'): #turn off dimensions measurements
            display_dimensions = False
        elif key & 0xFF == ord('f'):
            show_filtered_distance = True
        elif key & 0xFF == ord('g'):
            show_filtered_distance = False
        elif key & 0xFF == ord('a'):
            show_filtered = True
        elif key & 0xFF == ord('s'):
            show_filtered = False
            cv2.destroyWindow('Filtered')
        # elif key & 0xFF == ord('v'):
        #     camera.spatial_magnitude = 

        # except RuntimeError as e:
        #     print(f'{(e).__class__.__name__}: {e}') #error message
        #     continue

finally:
    camera.stop()