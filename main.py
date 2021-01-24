import cv2
from inference import colorization

img = cv2.imread("imgs/photo.jpeg")

output = colorization(img)

cv2.imshow("img", img)
cv2.imshow("output", output)
cv2.waitKey(0)
