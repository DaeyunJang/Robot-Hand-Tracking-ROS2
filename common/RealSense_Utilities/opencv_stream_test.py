import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")
import cv2

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("C:\\Users\\35840\\Documents\\20211217_204044.bag")
profile = pipe.start(cfg)

try:
    while True:
        # decimate = rs.decimation_filter(8)
        # align_to = rs.stream.color
        # align = rs.align(align_to)

        # # Get frameset of color and depth
        # frames = pipeline.wait_for_frames()
        # decimated = decimate.process(frames).as_frameset()
        # # Align the depth frame to color frame
        # aligned_frames = align.process(decimated)
        frameset = pipe.wait_for_frames()
        frame = frameset.get_depth_frame()

        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        colorizer = rs.colorizer()
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)

        temporal = rs.temporal_filter(0.4,40,8)
        hole_filling = rs.hole_filling_filter()

        # frame = depth_to_disparity.process(frame)
        # frame = spatial.process(frame)
        frame = temporal.process(frameset).as_frameset()
        # frame = frame.get_depth_frame()
        
        # frame = disparity_to_depth.process(frame)
        # frame = hole_filling.process(frame)
        align = rs.align(rs.stream.color)
        frame = align.process(frame)
        frame = frame.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

        cv2.imshow('o', colorized_depth)

        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipe.stop()
