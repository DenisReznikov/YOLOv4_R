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

    def __init__(self, config_file="cfg/yolov4-obj.cfg", data_file = "data/obj.data", weights = "custom-yolov4-detector_best.weights"):
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
        random.seed(3)  
        network, class_names, class_colors = darknet.load_network(
            self.__config_file,
            self.__data_file,
            self.__weights,
            batch_size=batch_size
        )

        prev_time = time.time()
        image, detections = self.__image_detection(
            image_rgb, network, class_names, class_colors, thresh
            )
        label = []
        confidence = []
        bbox = []

        for label_, confidence_, bbox_ in detections:
                bbox.append(self.__convert2relative(image, bbox_))
                label.append(label_)
                confidence.append(confidence_)
        return  label, confidence, bbox
