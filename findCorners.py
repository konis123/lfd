import numpy as np
import cv2
#%matplotlib inline

#코너들의 좌표를 리턴
def findCorners(img, nx, ny):
    #resultImage = np.copy(img)

    #엣지찾기
    mag_binary = mag_thresh(img, sobel_kernel=5, mag_thresh=(150, 255))

    rows = []
    cols = []
    corners = []

    lines = cv2.HoughLines(mag_binary, 1, np.pi/90, 20)
    for line in lines:
        r, theta = line[0]
        angle = int(theta*180/np.pi)
        if angle == 90:#가로줄
            x0 = 0
            y0 = r
            x1 = int(x0 + 1000*(-1))
            y1 = int(y0)
            x2 = int(x0 - 1000*(-1))
            y2 = int(y0)
            rows.append(r)
            #cv2.line(resultImage, (x1, y1), (x2, y2), (0, 255, 0), 1)
        elif angle == 0:#세로줄
            x0 = r
            y0 = 0
            x1 = int(x0)
            y1 = int(y0 + 1000)
            x2 = int(x0)
            y2 = int(y0 - 1000)
            cols.append(r)
            #cv2.line(resultImage, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cols.sort()
    rows.sort()
    print(cols)
    print(rows)


    #이제 코너 픽셀좌표 리턴하는걸 만들어야함
    for col in cols:
        for row in rows:
            corners.append((col,row))

    #print(corners)
    return corners

#코너를 찾은 이미지 리턴
def findCornersImage(img, nx, ny):
    resultImage = np.copy(img)

    #엣지찾기
    mag_binary = mag_thresh(img, sobel_kernel=5, mag_thresh=(150, 255))
    #cv2.imshow('img', mag_binary)

    #이거 일단 보류
    # lines = cv2.HoughLinesP(mag_binary, 1, np.pi/180, 200, 10, 100)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1,y1), (x2, y2), (0, 255, 0), 2)

    lines = cv2.HoughLines(mag_binary, 1, np.pi/90, 50)
    for line in lines:
        r, theta = line[0]
        angle = int(theta*180/np.pi)
        if angle == 90:#가로줄
            x0 = 0
            y0 = r
            x1 = int(x0 + 1000*(-1))
            y1 = int(y0)
            x2 = int(x0 - 1000*(-1))
            y2 = int(y0)
            cv2.line(resultImage, (x1, y1), (x2, y2), (0, 255, 0), 1)
        elif angle == 0:#세로줄
            x0 = r
            y0 = 0
            x1 = int(x0)
            y1 = int(y0 + 1000)
            x2 = int(x0)
            y2 = int(y0 - 1000)
            cv2.line(resultImage, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return resultImage


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    # Return the binary image
    return binary_output




img = cv2.imread('./bg.jpg')

rImg = findCornersImage(img, 8, 1)
cv2.imshow('d',rImg)

findCorners(img, 8, 1)


cv2.waitKey(0)
cv2.destroyAllWindows()
