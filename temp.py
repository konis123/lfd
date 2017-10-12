#아래 코드가 glob으로 여러각도와 거리에서 측정된 사진으로 캘리브레이션해서 보정하고 왜곡수정한 사진 보여주는거임 ㅎㅎ...
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
#%matplotlib inline

objpoints = []
imgpoints = []

img = cv2.imread('./bg.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (6,4), None)

print(int(corners[2][0][0]))

img = cv2.drawChessboardCorners(img, (6,4), corners, ret)
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#dst = cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imshow('undistorted image', dst)
cv2.imshow('img', img)
key = cv2.waitKey(0)
