import cv2
from darknet_images import classify_photo
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(classify_photo(image_rgb))