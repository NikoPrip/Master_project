import numpy as np
import cv2 as cv
import glob
import os

class CameraCalibrator:
    def __init__(self, images_dir='Images', chessboard_size=(8, 11), crop_percent=0.125):
        self.images_dir = images_dir
        self.chessboard_size = chessboard_size
        self.crop_percent = crop_percent
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def collect_image_points(self):
        images = glob.glob(os.path.join(self.images_dir, '*.jpeg'))
        print(f"Found {len(images)} images in {self.images_dir}.")
        im_count = 0
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                im_count += 1
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                cv.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                #cv.imshow('img', img)
                #cv.waitKey(1)
        #print(f"Successfully processed {im_count} images.")
        cv.destroyAllWindows()

    def calibrate(self, image_shape):
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, image_shape, None, None)
        print("Camera matrix:")
        print(self.mtx)
        print("Distortion coefficients:")
        print(self.dist)
        return ret

    def undistort_and_crop(self, image_path, output_path='calibresult.png'):
        img = cv.imread(image_path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
        crop_x = int(w * self.crop_percent)
        crop_y = int(h * self.crop_percent)
        dst_cropped = dst[crop_y:h-crop_y, crop_x:w-crop_x]
        cv.imwrite(output_path, dst_cropped)
        #print(f"Saved undistorted and cropped image to {output_path}")

    def compute_reprojection_error(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        total_error = mean_error / len(self.objpoints)
        #print(f"Total reprojection error: {total_error}")
        return total_error

if __name__ == "__main__":
    calibrator = CameraCalibrator(images_dir="/home/nikolai/Uni/Master_project/src/camera_calibration/Images", chessboard_size=(8, 11), crop_percent=0.125)
    calibrator.collect_image_points()
    # Use the shape of the first image for calibration
    first_image = glob.glob(os.path.join(calibrator.images_dir, '*.jpeg'))[0]
    img = cv.imread(first_image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    calibrator.calibrate(gray.shape[::-1])
    calibrator.undistort_and_crop(os.path.join(calibrator.images_dir, 'frame_00570.jpeg'))
    calibrator.compute_reprojection_error()