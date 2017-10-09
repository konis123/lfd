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


#아래 코드가 glob으로 여러각도와 거리에서 측정된 사진으로 캘리브레이션해서 보정하고 왜곡수정한 사진 보여주는거임 ㅎㅎ...
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
#%matplotlib inline


images = glob.glob('./calibration_wide/GOPR00*.jpg')

objpoints = []
imgpoints = []

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)


with open('calib.p','wb') as f:
    pickle.dump(objpoints, f)
    pickle.dump(imgpoints, f)
    f.close()


temp = cv2.imread('./calibration_wide/GOPR0032.jpg')

img = cv2.drawChessboardCorners(img, (8,6), corners, ret)
#plt.imshow(img)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
dst = cv2.undistort(temp, mtx, dist, None, mtx)
cv2.imshow('asdf', dst)
key = cv2.waitKey(0)
#if key==27:
#    cv2.destroyAllWindows()
#    break
#cv2.destroyAllWindows()
