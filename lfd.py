import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import math


#img = cv2.imread('./bg.jpg')

CAMERA_HEIGHT = 120#400
CHESSBOARD_DISTANCE = 300
CHESSBOARD_HEIGHT = 100


def getTheta(camera_height, chessboard_distance, chessboard_height, radian=True):

    temp = samePhaseDist(camera_height, chessboard_distance, chessboard_height)

    if radian == True:
        return math.atan(camera_height/temp)
    else:
        return math.atan(camera_height/temp) * 180/math.pi

def samePhaseDist(camera_height, chessboard_distance, chessboard_height):
    phase_dist = ( chessboard_distance * chessboard_height / (camera_height - chessboard_height) ) + chessboard_distance

    return phase_dist


temp = samePhaseDist(CAMERA_HEIGHT, CHESSBOARD_DISTANCE, CHESSBOARD_HEIGHT)

print('체스보드 중앙과 같은위상의 바닥까지의 거리 : ', temp)

print('theta : ', getTheta(CAMERA_HEIGHT, CHESSBOARD_DISTANCE, CHESSBOARD_HEIGHT))




img = cv2.imread('bg.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (6,4), None)
'''
print(int(corners[12][0][0]))#왼쪽
print(int(corners[12][0][1]))

print(int(corners[17][0][0]))#오른쪽
print(int(corners[17][0][1]))
'''
che_dis = int(corners[17][0][0]) - int(corners[12][0][0])
print(che_dis)


img = cv2.drawChessboardCorners(img, (6,4), corners, ret)
