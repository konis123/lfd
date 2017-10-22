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



if __name__=="__main__":
    obj = input('몇 m 거리를 표시할까요?')
    obj = int(obj)*100

    imgFilePath = input('이미지 경로 입력 : ')
    img = cv2.imread('./iphone3_white.jpeg')#cv2.imread(imgFilePath)

    ### 알고있는 정보이므로 직접 입력해주어야함
    CAMERA_HEIGHT = 160
    CAMERATOBONNET = 100
    BONNETTOCHESS = 100
    CHESS_SPACE = 20
    CHESS_HEIGHT = 100
    CAMERATOCHESS = CAMERATOBONNET + BONNETTOCHESS
    ###

    # 가로, 세로 반환
    img_size = img.shape[:2]

    #체스판 코너검출 및 그리기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (15,5), None)
    img = cv2.drawChessboardCorners(img, (15,5), corners, ret)

    #체스보드를 이용해서 소실선의 픽셀위치(banishingLine)를 찾음
    CHESS_SPACE_PIXEL = abs(corners[7][0][1]-corners[22][0][1]) #corners[0][0][0]: 가로 corners[0][0][1]: 세로
    a = (CAMERA_HEIGHT - CHESS_HEIGHT)/CHESS_SPACE  #소실선과 체스판 밑면사이에 몇개의 칸이 있는지 구함
    banishingLine = corners[67][0][1] - CHESS_SPACE_PIXEL*(a-1) #코너가 체스보드 맨밑에서시작이아니라 한칸위에서시작하므로 하나뺸만큼 빼줘야함
    cv2.line(img, (0,int(banishingLine)), (img_size[0],int(banishingLine)), (255,0,0), 5)
    cv2.circle(img, (corners[67][0][0], int(banishingLine)), 10, (0,255,0), -1)


    #체스보드와 카메라와의 각도 cTheta
    cTheta = math.atan(CAMERA_HEIGHT / CAMERATOCHESS)
    cAngle = cTheta * 180/math.pi
    print('체스보드와 카메라와의 각도 : ',cAngle,'도')

    # 1도당 나태내는 픽셀수. 체스판한칸*8은 바닥에서 체스판가운데까지의 픽셀수
    pa = CHESS_SPACE_PIXEL*8/cAngle

    #input으로 받은 값과 소실점의 픽셀거리 차이구하기
    oTheta = math.atan(CAMERA_HEIGHT/obj)
    oAngle = oTheta * 180/math.pi
    print('구하고자하는 거리와 카메라와의 각도 : ',oAngle,'도')

    # 구하고자하는 라인의 x픽셀 위치
    objLine = banishingLine + oAngle*pa
    cv2.line(img, (0,int(objLine)), (img_size[0],int(objLine)), (0,0,255), 5)



    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()