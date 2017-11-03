'''
import cv2
import numpy as np

if __name__ == '__main__':
    # Read source image.
    im_src = cv2.imread('book2.jpg')
    # Four corners of the book in source image
    pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])

    # Read destination image.
    im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)

    cv2.waitKey(0)

'''


'''
#마커와의 위치를 알기위한 이미지가 하나 필요하므로 과제와는 약간 다른주제임...ㅠㅠ 뭘찾아야하는거지
import numpy as np
import cv2

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	print(cv2.minAreaRect(c))
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0 * 2.54

# initialize the known object width, which in this case, the piece of
# paper is 12 cm wide
KNOWN_WIDTH = 11.0 * 2.54

# initialize the list of images that we'll be using
IMAGE_PATHS = ["bg.jpg"]
# load the first image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for imagePath in IMAGE_PATHS:
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	marker = find_marker(image)
	cm = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

	# draw a bounding box around the image and display it
	box = np.int0(cv2.boxPoints(marker))
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fcm" % (cm / 12),
		(image.shape[1] - 240, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

def calibrateAndSave(nx, ny, imagesPath=None):
    objpoints = []
    imgpoints = []
    dist_pickle = {}


    images = glob.glob(imagesPath)#'./calibration_wide/GOPR00*.jpg'
    tempImage = mpimg.imread(images[0])

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow('gray',gray)
        #cv2.waitKey(0)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        ###임시적으로 코너그린거보려고 넣은거임#######################
        '''print(ret)
        temp = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        cv2.imshow('img',temp)'''
        ###

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    if len(imgpoints) == 0:
        print('corners 발견하지 못함')
        return

    with open('calib.p', 'wb') as f:
        dist_pickle['objpoints'] = objpoints
        dist_pickle['imgpoints'] = imgpoints
        pickle.dump(dist_pickle, f)
        f.close()


    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('tvecs : ', tvecs)

    with open('calib.p', 'wb') as f:
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump(dist_pickle, f)
        f.close()

    return mtx, dist

#testing
def corners_unwarp(img, mtx, dist, distorted=False, corners=None, nx=None, ny=None):
    if distorted:
        img = cv2.undistort(img, mtx, dist, None, mtx)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Xoffset = 100.0  # offset for dst points
    Yoffset = 300.0
    offset = 000

    #img_size[0] : 가로
    #img_size[1] : 세로
    img_size = (gray.shape[1], gray.shape[0])

    src = np.float32([[corners[0][0][0] - Xoffset, corners[0][0][1] - Yoffset],
                      [corners[nx - 1][0][0] + Xoffset, corners[nx - 1][0][1] - Yoffset],
                      [corners[-1][0][0] + Xoffset, corners[-1][0][1] + Yoffset],
                      [corners[-nx][0][0] - Xoffset, corners[-nx][0][1] + Yoffset]])

    #src = np.float32([corners[0], corners[1], corners[-1], corners[2]])

    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                 [img_size[0]-offset, img_size[1]-offset],
                                 [offset, img_size[1]-offset]])


    #src = np.float32([[0,0],[img_size[0],0],[img_size[0],img_size[1]],[0,img_size[1]])
    #dst = np.float32([])
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(undist, M, img_size)

    return warped, M


#현재 이거
def corners_unwarp2(img, nx, ny, mtx, dist, points=None):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    #print(corners.shape,"dsadsf")
    M = 0
    warped = 0

    cv2.imshow('undist',img)

    print(ret)
    if ret == True:
        # If we found corners, draw them! (just for fun)
        #####cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exacWt, but close enough for our purpose here
        Xoffset = 500#200.0 # offset for dst points
        Yoffset = 500#300.0
        offset = 0
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])
        print(corners[0][0][0])
        # For source points I'm grabbing the outer four detected corners
        #src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        src = np.float32([ [corners[0][0][0]-Xoffset,corners[0][0][1]-Yoffset], [corners[nx-1][0][0]+Xoffset, corners[nx-1][0][1]-Yoffset], [corners[-1][0][0]+Xoffset, corners[-1][0][1]+Yoffset], [corners[-nx][0][0]-Xoffset, corners[-nx][0][1]+Yoffset] ])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


if __name__=="__main__":

    distImage = cv2.imread('./iphone3.jpeg')
    mtx, dist = calibrateAndSave(nx=14, ny=4, imagesPath='./iphone3.jpeg')
    #distImage = cv2.imread('./calibration_wide/GOPR004.jpg')
    #mtx, dist = calibrateAndSave(nx=8, ny=6, imagesPath='./calibration_wide/GOPR0034.jpg')

    #mtx, dist를 이용해 왜곡을 수정한 이미지를 구함
    undistorted = cv2.undistort(distImage, mtx, dist, None, mtx)
    cv2.imshow('undistort', undistorted)


    topdownView, M = corners_unwarp2(distImage, nx=14, ny=4, mtx=mtx, dist=dist, points=None)
    #topdownView, M = corners_unwarp2(distImage, nx=8, ny=6, mtx=mtx, dist=dist, points=None)
    cv2.imshow('asdf', topdownView)





    cv2.waitKey(0)
    cv2.destroyAllWindows()
