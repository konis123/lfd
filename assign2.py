import cv2
import numpy as np
import math


'''
우리가 알고있는 정보 (단위: cm)

카메라 높이 : 160
카메라에서 보닛까지의 거리 : 100
보닛에서 체스보드까지와의 거리 : 100
체스보드한칸의 길이 : 20
체스보드 밑변과 바닥까지의 거리 : 100
'''


def readData():

    f = open('data.txt','r')
    lines = f.readlines()
    data = []
    for line in lines:
        s = line.split('=')
        data.append(float(s[1]))

    return data[0], data[1], data[2], data[3], data[4], int(data[5]), int(data[6])


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #거리구하기
        temp = y - banishingLine    #소실선과 y좌표 픽셀차
        clickedAngle = temp/pa   #이게 우리가 찍은 바닥에서 카메라와의 각도
        objZ = CAMERA_HEIGHT / math.tan(clickedAngle * math.pi/180)

        b2c = getBansishingLine2Chess()

        #8은 가운데에서 가로로 8칸이있다는거
        pd = (8 * CHESS_SPACE) / (corners[Y_CORNERS*X_CORNERS-1][0][0]-corners[Y_CORNERS*X_CORNERS - (Y_CORNERS//2) - 1][0][0])  #체스판에서의 픽셀당 센치(가운데에서 수평방향으로)
        newpd = b2c / temp * pd #클릭한곳에서의 픽셀당 센치미터
        objX = newpd * (x-corners[Y_CORNERS*X_CORNERS - (Y_CORNERS//2) - 1][0][0])
        print( '수직방향거리: ',objZ,'수평방향거리: ',objX )  #픽셀당 센치에서 클릭한곳의 좌표만큼 곱해준거

#소실점에서 체스판 바닥까지의 픽셀차이 구하기
def getBansishingLine2Chess():

    #체스판이 있는곳의 픽셀위치구하기
    oTheta = math.atan(CAMERA_HEIGHT/CAMERATOCHESS)
    oAngle = oTheta * 180/math.pi
    objLine = banishingLine + oAngle*pa

    #체스판이 있는 x픽셀 위치에서 소실선 x픽셀 위치를 뺸값을 리턴
    return objLine - banishingLine

#global CAMERA_HEIGHT, CAMERATOBONNET, BONNETTOCHESS, CHESS_SPACE, CHESS_HEIGHT, CAMERATOCHESS
#global banishingLine

if __name__=="__main__":
    obj = input('몇 m 거리를 표시할까요?')
    obj = float(obj)*100

    imgFilePath = input('이미지 경로 입력 : ')
    #img = cv2.imread('./iphone3_white0.jpg')
    img = cv2.imread(imgFilePath)

    ### 알고있는 정보이므로 직접 입력해주어야함
    '''
    CAMERA_HEIGHT = 140
    CAMERATOBONNET = 0
    BONNETTOCHESS = 150
    CHESS_SPACE = 10
    CHESS_HEIGHT = 120
    X_CORNERS = 5
    Y_CORNERS = 15
    CAMERATOCHESS = CAMERATOBONNET + BONNETTOCHESS
    '''
    ###

    #data.txt 파일에서 위의 알고있는 값들을 읽어옴
    CAMERA_HEIGHT, CAMERATOBONNET, BONNETTOCHESS, CHESS_SPACE, CHESS_HEIGHT, X_CORNERS, Y_CORNERS = readData()
    CAMERATOCHESS = CAMERATOBONNET + BONNETTOCHESS

    # 가로, 세로 반환
    img_size = img.shape[:2]

    #체스판 코너검출 및 그리기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (Y_CORNERS, X_CORNERS), None)
    img = cv2.drawChessboardCorners(img, (Y_CORNERS,X_CORNERS), corners, ret)

    #체스보드를 이용해서 소실선의 픽셀위치(banishingLine)를 찾음
    CHESS_SPACE_PIXEL = abs(corners[(Y_CORNERS//2)-1][0][1]-corners[(Y_CORNERS//2)+Y_CORNERS-1][0][1]) #corners[0][0][0]: 가로 corners[0][0][1]: 세로
    a = (CAMERA_HEIGHT - CHESS_HEIGHT)/CHESS_SPACE  #소실선과 체스판 밑면사이에 몇개의 칸이 있는지 구함
    banishingLine = corners[Y_CORNERS*X_CORNERS - (Y_CORNERS//2) - 1][0][1] - CHESS_SPACE_PIXEL*(a-1) #코너가 체스보드 맨밑에서시작이아니라 한칸위에서시작하므로 하나뺸만큼 빼줘야함
    cv2.line(img, (0,int(banishingLine)), (img_size[0],int(banishingLine)), (255,0,0), 5)
    cv2.circle(img, (corners[Y_CORNERS*X_CORNERS - (Y_CORNERS//2) - 1][0][0], int(banishingLine)), 10, (0,255,0), -1)


    #체스보드와 카메라와의 각도 cTheta
    cTheta = math.atan((CAMERA_HEIGHT - CHESS_HEIGHT) / CAMERATOCHESS)#
    cAngle = cTheta * 180/math.pi
    print('체스보드와 카메라와의 각도 : ',cAngle,'도')

    # 1도당 나태내는 픽셀수. 체스판한칸*8은 바닥에서 체스판가운데까지의 픽셀수
    pa = (corners[Y_CORNERS*X_CORNERS - (Y_CORNERS//2) - 1][0][1] + CHESS_SPACE_PIXEL - banishingLine) / cAngle

    #input으로 받은 값과 소실점의 픽셀거리 차이구하기
    oTheta = math.atan(CAMERA_HEIGHT/obj)
    oAngle = oTheta * 180/math.pi
    print('구하고자하는 거리와 카메라와의 각도 : ',oAngle,'도')

    # 구하고자하는 라인의 x픽셀 위치
    objLine = banishingLine + oAngle*pa
    cv2.line(img, (0,int(objLine)), (img_size[0],int(objLine)), (0,0,255), 5)


    #이미지 띄우기
    cv2.imshow('img', img)

    #마우스 콜백 등록
    cv2.setMouseCallback('img', onMouse, param=img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
