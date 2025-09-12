import numpy as np
import cv2
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description='undistort images, if no arguments are given, /input and /output in current folder are used')
    parser.add_argument('-i', '--input', metavar='', type=lambda p: Path(p).absolute(), help='the input directory', default=Path(__file__).absolute().parent / "input")
    parser.add_argument('-o', '--output', metavar='', type=lambda p: Path(p).absolute(), help='the output directory', default=Path(__file__).absolute().parent / "output")
    parser.add_argument('-c', '--calibration', metavar='', type=lambda p: Path(p).absolute(), help='path to the calibration file', default=None)

    args = parser.parse_args()

    # make sure output directory exists, otherwise we make it
    args.output.mkdir(parents=False, exist_ok=True)
    # import path names of all images
    list_input = generate_list_of_images(args.input)
    # grab calibration file
    calibration_file = get_calibration_file(args.calibration, args.input)
    # undistort images
    undistort_images(list_input, args.output, calibration_file)


def undistort_images(list_input, output, calibration_file):
    f = open(str(calibration_file), "r")
    file = f.readlines()
    # fx fy cx cy
    m = file[2].split()
    # k1 k2 k3 k4   or   k1 k2 p1 p2 k3
    d = file[5].split()
    f.close()

    matrix = np.array([[m[0], 0, m[2]],
                       [0, m[1], m[3]],
                       [0, 0, 1]]).astype(float)
    if len(d) == 4:  # fisheye model
        distortion = np.array([[d[0]],
                                [d[1]],
                                [d[2]],
                                [d[3]]]).astype(float)

        for fname in list_input:
            img = cv2.imread(str(fname))
            h,  w = img.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(matrix, distortion, np.eye(3), matrix, (w, h), cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), undistorted_img)
    elif len(d) == 5:
        distortion = np.array([[d[0]],
                               [d[1]],
                               [d[2]],
                               [d[3]],
                               [d[4]]]).astype(float)

        for fname in list_input:
            img = cv2.imread(str(fname))
            h,  w = img.shape[:2]
            newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
            dst = cv2.undistort(img, matrix, distortion, None, newcamera_mtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), dst)
    else:
        sys.exit("Something is wrong with the calibration file... Redo calibration and try again")




def generate_list_of_images(path_to_dir):
    """
    Returns a list of all the image paths in the provided directory
    """
    assert(path_to_dir.is_dir())
    file_paths_input = []
    for file in path_to_dir.iterdir():
        if file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            file_paths_input.append(file)
    return file_paths_input

def get_calibration_file(path_to_calibration, path_to_dir):
    if path_to_calibration is not None:
        return path_to_calibration
    else:
        for file in path_to_dir.glob('calibration_*'):
            return file
    sys.exit("No calibration file found")


if __name__ == '__main__':
    main()









