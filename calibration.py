import numpy as np
import cv2 as cv
import glob
import os

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)

objpoints = []
imgpoints = []

images_dir = os.path.join(os.path.dirname(__file__), 'Images')
images = glob.glob(os.path.join(images_dir, '*.jpeg'))
print(f"Found {len(images)} images in {images_dir}.")

im_count = 0

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (8,11), None)

    if ret == True:
        im_count += 1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, (8,11), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1)
print(f"Successfully processed {im_count} images.")
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print("Camera matrix:")
#print(mtx)

# Undistort an example image
img = cv.imread("Images/frame_00570.jpeg")
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
 
# crop the image
x, y, w, h = roi
x = max(x - 20, 0)
w = min(w + 40, dst.shape[1] - x)
dst_cropped = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst_cropped)