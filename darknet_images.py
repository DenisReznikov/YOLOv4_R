import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet


class Yolo_wrapper():

    def __init__(self, logger, config_file="cfg/yolov4-obj.cfg", data_file = "data/obj.data", weights = "custom-yolov4-detector_best.weights"):
        self.__logger = logger
        self.__config_file = config_file
        self.__data_file = data_file
        self.__weights = weights
        
    

    def __image_detection(self, image, network, class_names, class_colors, thresh):
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)
        
        image_rgb = image
        image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


    def __convert2relative(self, image, bbox):
        x, y, w, h = bbox
        height, width, _ = image.shape
        return x/width, y/height, w/width, h/height


    def __save_annotations(self, name, image, detections, class_names):
        file_name = name.split(".")[:-1][0] + ".txt"
        with open(file_name, "w") as f:
            for label, confidence, bbox in detections:
                x, y, w, h = self.__convert2relative(image, bbox)
                label = class_names.index(label)
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


    def classify_photo(self, image_rgb, thresh=0.25, batch_size=1):
        
        if not os.path.exists(self.__config_file):
            self.__logger.error(f'Load file error. Config_file')
            raise(ValueError("Invalid config path {}".format(os.path.abspath(self.__config_file))))
        if not os.path.exists(self.__weights):
            self.__logger.error(f'Load file error. Weights')
            raise(ValueError("Invalid weight path {}".format(os.path.abspath(self.__weights))))
        if not os.path.exists(self.__data_file):
            self.__logger.error(f'Load file error. Data_file')
            raise(ValueError("Invalid data file path {}".format(os.path.abspath(self.__data_file))))


        if (len(set(image_rgb.shape)) > 2):
            self.__logger.error(f'Image dont have correct shape')
            raise ValueError("Image must have correct shape")
                
        random.seed(3)  
        prev_time = time.time()

        network, class_names, class_colors = darknet.load_network(
            self.__config_file,
            self.__data_file,
            self.__weights,
            batch_size=batch_size
        )

        try:
            image, detections = self.__image_detection(
                image_rgb, network, class_names, class_colors, thresh
                )
        except Exception:
            self.__logger.error(f'Error in yolo.In C path')

        speed = str(time.time() - prev_time)
        self.__logger.info('Time for one predict: ' + (speed))



        label = []
        confidence = []
        bbox = []

        for label_, confidence_, bbox_ in detections:
                bbox.append(self.__convert2relative(image, bbox_))
                label.append(label_)
                confidence.append(confidence_)
        return  label, confidence, bbox
