'''Program is using this code with modifications: 
https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/distance_to_object.ipynb'''
# TODO: 
#   - 
#   - 
#   - 
#   - 


import cv2

# Color constant for opencv
WHITE = (255, 255, 255)

class ObjectDetect:

    def __init__(self, color_frame, depth_frame, depth_scale):
        self.color_frame = color_frame
        self.depth_frame = depth_frame
        self.depth_scale = depth_scale
        #self.color_intrinsics = color_intrin

        self.confidence: float

        # Rectangle that is drawn around the detected object
        self.rectangle_xmin = None
        self.rectangle_ymin = None
        self.rectangle_xmax = None
        self.rectangle_ymax = None


    def detect(self):
        # Standard OpenCV boilerplate for running the net:
        height, width = self.color_frame.shape[:2]
        expected = 300
        aspect = width / height
        resized_image = cv2.resize(self.color_frame, (round(expected * aspect), expected))
        crop_start = round(expected * (aspect - 1) / 2)
        crop_img = resized_image[0:expected, crop_start:crop_start+expected]

        net = cv2.dnn.readNetFromCaffe("C:\\Users\\35840\\Downloads\\MobileNetSSD\\MobileNetSSD\\MobileNetSSD_deploy.prototxt", "C:\\Users\\35840\\Downloads\\MobileNetSSD\\MobileNetSSD\\MobileNetSSD_deploy.caffemodel")
        inScaleFactor = 0.007843
        meanVal       = 127.53
        class_names = ("background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair",
                    "cow", "diningtable", "dog", "horse",
                    "motorbike", "person", "pottedplant",
                    "sheep", "sofa", "train", "tvmonitor")
        #class_names = ("bottle", "chair", "diningtable", "person", "pottedplant", "tvmonitor")
        blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
        net.setInput(blob, "data")
        detections = net.forward("detection_out")

        label = detections[0,0,0,1]
        conf  = detections[0,0,0,2]
        xmin  = detections[0,0,0,3]
        ymin  = detections[0,0,0,4]
        xmax  = detections[0,0,0,5]
        ymax  = detections[0,0,0,6]

        class_name = class_names[int(label)]

        confidence =  str(round(conf,2))[0:4]

        # cv2.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)), 
        #             (int(xmax * expected), int(ymax * expected)), (255, 255, 255), 2)

        scale = height / expected
        xmin_depth = int((xmin * expected + crop_start) * scale)
        ymin_depth = int((ymin * expected) * scale)
        xmax_depth = int((xmax * expected + crop_start) * scale)
        ymax_depth = int((ymax * expected) * scale)

        # Crop depth data:
        depth = self.depth_frame[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)

        # Get data scale from the device and convert to meters
        depth = depth * self.depth_scale
        distance,_,_,_ = cv2.mean(depth)
        
        # if class_name not in ["bottle", "chair", "diningtable", "person", "pottedplant", "tvmonitor"]:
        #     class_name = "unknown"
                
        self.rectangle_xmin = xmin_depth
        self.rectangle_ymin = ymin_depth
        self.rectangle_xmax = xmax_depth
        self.rectangle_ymax = ymax_depth
        self.confidence = confidence
        self.class_name = class_name
        self.distance = distance

    def draw_rectangle(self, image):
        
        # Rectangle variables 
        top_left_corner = (self.rectangle_xmin, self.rectangle_ymin)
        bottom_right_corner = (self.rectangle_xmax, self.rectangle_ymax)
        rectangle_color = WHITE
        rectangle_thickness = 2

        # Text variables
        text = f'{self.class_name} {self.distance:.2f} meters away'
        text_location = (self.rectangle_xmin, self.rectangle_ymin - 5)
        text_font = cv2.FONT_HERSHEY_COMPLEX
        text_color = WHITE
        text_scale = 0.5

        cv2.rectangle(image,
                      top_left_corner, bottom_right_corner,
                      rectangle_color, rectangle_thickness)

        cv2.putText(self.color_frame, text,
                    text_location, text_font,
                    text_scale, text_color)