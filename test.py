import logging
import cv2
from darknet_images import Yolo_wrapper
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

yolo = Yolo_wrapper(logger=logging)
label, confidence, bbox = (yolo.classify_photo(image_rgb))
print(label, confidence, bbox)