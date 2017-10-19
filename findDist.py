import numpy as np
import cv2
import math


def getDistance(ppo, y, banishingLineY, camera_height):
    temp = y - banishingLineY
    clickedAngle = temp/ppo   #이게 우리가 찍은 바닥에서 카메라와의 각도
    clickedDistance = camera_height / math.tan(clickedAngle * math.pi/180)
    print(clickedDistance,'cm')

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        temp = y - banishingLineY
        print(x, ', ', y, ' clicked \n distance(소실선에서 y좌표까지 픽셀거리) : ', temp)
        clickedAngle = temp/ppo   #이게 우리가 찍은 바닥에서 카메라와의 각도
        clickedDistance = camera_height / math.tan(clickedAngle * math.pi/180)
        print(clickedDistance,'cm')

        ##여기서부턴 x실제 길이를 구하려고한다
        temp2 = abs(banishingLineX - x)
        if clickedDistance < obj_dist:
            print('x거리:',(temp2)*obj_dist/clickedDistance)

        else:
            print('x거리:',(temp2)*obj_dist/clickedDistance)





obj_dist = 300  #####값조정이 필요함
obj_height = 120    #####
camera_height = 120 #####

img = cv2.imread('IMG4650.jpg')#IMG4650

height, width = img.shape[:2]
print(height,',', width)

banishingLineX = width/2
banishingLineY = height/2

cv2.imshow('original', img)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (15,6), None)
img = cv2.drawChessboardCorners(img, (15,6), corners, ret)

che_dis_one = int(corners[2][0][0]) - int(corners[3][0][0])
che_dis = che_dis_one * 20 ##사진에서 체스판의 한칸의 실제길이가 6cm로 가정
print(che_dis,"sdf")

cv2.imshow('chess', img)

cv2.setMouseCallback('chess', onMouse, param=img)


theta = math.atan(camera_height / obj_dist)
angle = theta * 180/math.pi
print(angle,' 도임 체스보드랑 카메라랑 각도')

ppo = che_dis/angle#che_dis/angle

print(ppo)

cv2.waitKey(0)
cv2.destroyAllWindows()
