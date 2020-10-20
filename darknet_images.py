import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
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


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))



def classify_photo(image_rgb,thresh=0.25):
    config_file__ = "cfg/yolov4-obj.cfg"
    data_file__ = "data/obj.data"
    weights__ = "custom-yolov4-detector_best.weights"
    batch_size__ = 1
    random.seed(3)  
    network, class_names, class_colors = darknet.load_network(
        config_file__,
        data_file__,
        weights__,
        batch_size=batch_size__
    )

    prev_time = time.time()
    image, detections = image_detection(
        image_rgb, network, class_names, class_colors, thresh
        )
    print(detections)
    #save_annotations(image_name, image, detections, class_names)
    #darknet.print_detections(detections, args.ext_output)
    #fps = int(1/(time.time() - prev_time))
    label = []
    confidence = []
    bbox = []

    for label_, confidence_, bbox_ in detections:
            bbox.append(convert2relative(image, bbox_))
            label.append(class_names.index(label_))
            confidence.append(confidence_)
    return  label, confidence, bbox
