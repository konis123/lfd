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
        print(len(images))
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)


    with open('calib.p', 'wb') as f:
        dist_pickle['objpoints'] = objpoints
        dist_pickle['imgpoints'] = imgpoints
        pickle.dump(dist_pickle, f)
        f.close()

    # if calib==True:#캘리브레이션진행?
    #     if imagesPath == None:
    #         print('no imagesPath')
    #         return
    #
    #     images = glob.glob(imagesPath)#'./calibration_wide/GOPR00*.jpg'
    #
    #     objp = np.zeros((ny*nx,3), np.float32)
    #     objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    #
    #     for fname in images:
    #         print(len(images))
    #         img = mpimg.imread(fname)
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    #
    #         if ret == True:
    #             imgpoints.append(corners)
    #             objpoints.append(objp)
    #
    #
    #     with open('calib.p', 'wb') as f:
    #         dist_pickle['objpoints'] = objpoints
    #         dist_pickle['imgpoints'] = imgpoints
    #         pickle.dump(dist_pickle, f)
    #         f.close()
    #
    # else:
    #     with open( "calib.p", "rb" ) as f:
    #         dist_pickle = pickle.load(f)
    #         objpoints = dist_pickle['objpoints']
    #         imgpoints = dist_pickle['imgpoints']
    #         f.close()

    #gray = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY)
    #ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    #img = cv2.drawChessboardCorners(dimg, (8,6), corners, ret)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    with open('calib.p', 'wb') as f:
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump(dist_pickle, f)
        f.close()

    return mtx, dist

    #왜곡을 보정한 이미지를 구하고 리턴함
    #print(dist)
    #undist = cv2.undistort(dimg, mtx, dist, None, mtx)
    #return undist

    ####탑다운뷰와 변환하는 매트릭스를 구함 undist가아니라 dimg였음
    # topdown, perspective_M = corners_unwarp(dimg, nx, ny, mtx, dist)
    # print(perspective_M)
    #
    # return topdown



#현재 이거
def corners_unwarp(img, nx, ny, mtx, dist, points=None):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    #print(corners.shape,"dsadsf")
    M = 0
    warped = 0

    if ret == True:
        # If we found corners, draw them! (just for fun)
        #####cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
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



distImage = cv2.imread('./calibration_wide/GOPR0032.jpg')
mtx, dist = calibrateAndSave(nx=8, ny=6, imagesPath='./calibration_wide/*.jpg')

topdownView, M = corners_unwarp(distImage, nx=8, ny=6, mtx=mtx, dist=dist, points=None)
cv2.imshow('asdf', topdownView)





cv2.waitKey(0)
cv2.destroyAllWindows()
