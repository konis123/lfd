#거리구하기 과제를위한 방법
#방법1 직접적인 기하학적인 계산방법 
#방법2 호모그래피를 이용한 방법 은 실측사진을 이용해서 구하는것이므로(실측길이를 알수없다 현재 내가...), 내가 과제를해결하기위해서는 다른방법이 필요함
#방법3 3d변환을 이용한 방법

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

    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('img',img)
#cv2.imshow('idfds', edges)


nx = 6
ny = 4
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    cv2.imshow('chessSearch', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
