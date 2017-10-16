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




#tan안씌우고 비례한거
'''
aP = 110    #체스판 윗부분
bP = 90     #체스판 아랜부분

atobPixel = 30

tanAlpha = CHESSBOARD_DISTANCE / (CAMERA_HEIGHT - aP)
tanBeta = CHESSBOARD_DISTANCE / (CAMERA_HEIGHT - bP)

tanTheta = (tanAlpha - tanBeta) / (1 + tanAlpha*tanBeta)
theta = math.atan(tanTheta)

radPerPixel = theta / atobPixel

imgHeight = np.size(image, 0)
imgWidth = np.size(image, 1)

myPointY = 800  #임시로 아무값이나 넣어본거
vanishingPointY = imgHeight/2

myTheta = (somPointY - vanishingPointY) * radPerPixel

myDistance = CAMERA_HEIGHT / math.tan(myTheta)

print(myDistance)
'''

img = cv2.imread('nega.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (15,6), None)
if ret==True:

    che_dis = corners[30][0][1] - corners[45][0][1]#int(corners[30][0][1]) - int(corners[45][0][1])
    print(che_dis)


    img = cv2.drawChessboardCorners(img, (15,6), corners, ret)

    cv2.imshow('esf',img)
    cv2.waitKey(0)

    theta = math.atan(CAMERA_HEIGHT / CHESSBOARD_DISTANCE)
    temp = theta * 180 / math.pi;
    print(temp)

    oneDo = che_dis / temp
    print(oneDo)

else:
    print('why!?!?!??!?!?')
