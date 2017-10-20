import numpy as np
import cv2
import math
from iphone6s_calibration import calibrateAndSave
from iphone6s_calibration import corners_unwarp2
#수직방향향이 X 수평방향이 Y



#카메라높이가 주어지고, tilt가 0일떄, 100cm앞에있는 체스보드에서 소실점찾기
def findBanishingLine_100(img, ch):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (14,4), None)    #35

    #카메라높이 160일떄 소실선 위치
    banishingLineX = corners[35][0][1]
    print(corners[0])
    print(corners[35])
    interConers = corners[49][0][1] - corners[35][0][1]

    banishingLineX = banishingLineX + (ch-160)*interConers/20

    print('소실선 x좌표:', banishingLineX)

    #소실선의 x픽셀 좌표를 반환함
    return banishingLineX


def getDistance(ppo, y, banishingLineY, camera_height):
    temp = y - banishingLineY
    clickedAngle = temp/ppo   #이게 우리가 찍은 바닥에서 카메라와의 각도
    clickedDistance = camera_height / math.tan(clickedAngle * math.pi/180)
    print(clickedDistance,'cm')

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        temp = y - banishingLineX
        print(x, ', ', y, ' clicked \n distance(소실선에서 y좌표까지 픽셀거리) : ', temp)
        clickedAngle = temp/ppo   #이게 우리가 찍은 바닥에서 카메라와의 각도
        clickedDistance = camera_height / math.tan(clickedAngle * math.pi/180)
        print(clickedDistance,'cm')

        ##여기서부턴 x실제 길이를 구하려고한다
        '''
        temp2 = abs(banishingLineX - x)
        if clickedDistance < obj_dist:
            print('x거리:',(temp2)*obj_dist/clickedDistance)

        else:
            print('x거리:',(temp2)*obj_dist/clickedDistance)'''





obj_dist = 200  #####값조정이 필요함
obj_height = 160    #####
camera_height = 160 #####

if __name__=="__main__":

    img = cv2.imread('iphone3.jpeg')#IMG4650
    img2 = img.copy()

    height, width = img.shape[:2]
    print(height,',', width)

    banishingLineX = height/2
    #banishingLineY = width/2

    #cv2.imshow('original', img)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (14,4), None)
    img = cv2.drawChessboardCorners(img, (14,4), corners, ret)



    mtx, dist = calibrateAndSave(nx=14,ny=4,imagesPath='./iphone3.jpeg')
    undistorted = cv2.undistort(img2, mtx, dist, None, mtx)
    cv2.imshow('undistort', undistorted)
    topdownView, M = corners_unwarp2(img2, nx=14, ny=4, mtx=mtx, dist=dist, points=None)
    cv2.imshow('topdownView', topdownView)
    print(M)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
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
    '''
    print('========================')
    #findBanishingLine_100(img2, 161)
