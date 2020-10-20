import cv2
from darknet_images import classify_photo
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
label, confidence, bbox = (classify_photo(image_rgb))
print(label, confidence, bbox)