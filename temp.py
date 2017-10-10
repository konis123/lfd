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
'''
 #여기는 캘리브레이션 할때만 하면되고 보통 저장한값 불러쓰면댐
images = glob.glob('./calibration_wide/GOPR00*.jpg')

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)


with open('calib_objpoints.p','wb') as f:
    pickle.dump(objpoints, f)
    f.close()

with open('calib_imgpoints.p','wb') as f:
    pickle.dump(imgpoints, f)
    f.close()
'''

#objpoints하고 imgpoints 가져오기
with open( "calib_objpoints.p", "rb" ) as f:
    objpoints =pickle.load(f)
    f.close()
with open( "calib_imgpoints.p", "rb" ) as f:
    imgpoints =pickle.load(f)
    f.close()

img = cv2.imread('./calibration_wide/GOPR0032.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imshow('undistorted image', dst)
key = cv2.waitKey(0)
