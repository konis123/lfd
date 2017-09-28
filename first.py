import cv2
import numpy as np

img = cv2.imread('bg.jpg', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img,200,220, apertureSize = 3);
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

for line in lines:
    r, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('img',img)
cv2.imshow('idfds', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
