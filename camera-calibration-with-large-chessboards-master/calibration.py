import numpy as np
import cv2
import argparse
from pathlib import Path
from corner_detector import ChessBoardCornerDetector
import math
import datetime
import exifread
from icecream import ic


def main():
    parser = argparse.ArgumentParser(description='Calibrate camera with multiple images, if no arguments are given, /input and /output in current folder are used')
    parser.add_argument('-i', '--input', metavar='', type=lambda p: Path(p).absolute(), help='the input directory', default=Path(__file__).absolute().parent / "input")
    parser.add_argument('-o', '--output', metavar='', type=lambda p: Path(p).absolute(), help='the output directory', default=Path(__file__).absolute().parent / "output")
    parser.add_argument('--min_coverage', metavar='', type=float, default=25, help='Minimum percentage coverage of the checkerboard in the image for image to be considered (default 25)')
    parser.add_argument('-f', '--fisheye', dest='fisheye', action='store_true', help='set if camera is fisheye')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='set to debug the program and get intermediate steps')
    parser.add_argument('--scaling_debug', metavar='', type=float, default=1, help='Scaling factor for the estimation for distortion while debugging (default 1)')
    parser.add_argument('--kernel_size', metavar='', type=int, default=45, help='The size of the marker detection kernel (default 45)')
    parser.add_argument('--distance_scale', metavar='', type=int, default=100, help='The size of the neighbourhood when calculating relative responses (default 100)')
    parser.add_argument('--threshold_level_relative', metavar='', type=float, default=0.5, help='The threshold level to use for segmenting the relative marker response image (default 0.5)')
    parser.add_argument('--distance_threshold', metavar='', type=float, default=0.1, help='Threshold for acceptable deviation between espected location and actual location of points when assigning real world coordinates (default 0.1)')

    args = parser.parse_args()
    objpoints, imgpoints, all_imgpoints = [], [], []  # Every element is the list of one image
    stats_before, coverage_images = [], []
    # make sure output directory exists, otherwise we make it
    args.output.mkdir(parents=False, exist_ok=True)
    # import path names of all images
    list_input = generate_list_of_images(args.input)
    if len(list_input) == 0:
        print("ERROR: No files found at the provided input location, program stopped")
        exit()
    # detect corners in every image
    for file_path in list_input:
        (objp, imgp, coverage, statistics) = detect_calibration_pattern_in_image(file_path, args.output, args.debug, 
                args.kernel_size, args.distance_scale, args.threshold_level_relative, args.distance_threshold)
        stats_before.append(statistics)
        coverage_images.append(coverage)
        all_imgpoints.append(imgp)
        if coverage < args.min_coverage:
            print("ERROR: Less than " + str(args.min_coverage) + "% is covered with points, excluding from calibration")
        else:
            objpoints.append(objp)
            imgpoints.append(imgp)
    # calibrate camera
    img = cv2.imread(str(list_input[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrix, distortion = calibrate_camera(gray.shape[::-1], objpoints, imgpoints, args.fisheye)
    # undistort images
    path_to_undistorted_images = args.output / '6_undistorted_images'
    path_to_undistorted_images.mkdir(parents=False, exist_ok=True)
    stats_after = undistort_images(list_input, path_to_undistorted_images, matrix, distortion, args.fisheye, args.debug)
    # print output to screen and file
    print_output(matrix, distortion, args.fisheye)
    write_output(list_input, args.output, matrix, distortion, coverage_images, stats_before, stats_after, args.min_coverage, args.fisheye)
    write_calibration(list_input, args.output, matrix, distortion, args.fisheye)
    if args.debug:
        path_to_estimation_distortion = args.output / '7_estimation_distortion'
        path_to_estimation_distortion.mkdir(parents=False, exist_ok=True)
        show_estimation_distortion(list_input, all_imgpoints, matrix, distortion, args.fisheye, path_to_estimation_distortion, args.scaling_debug)


def show_estimation_distortion(list_input, imgpoints, matrix, distortion, fisheye, output, scaling_debug):
    """
    Shows an estimation of the distortion based on the camera model
    """
    converted_imgpoints = convert_coordinates(imgpoints, matrix, distortion, fisheye)
    for count, fname in enumerate(list_input):
        img = cv2.imread(str(fname))
        for imgpoint, conv_imgpoint in zip(imgpoints[count],converted_imgpoints[count]):
            x1 = int(imgpoint[0])
            y1 = int(imgpoint[1])
            x2 = int(x1 + (conv_imgpoint[0] - x1) * scaling_debug)
            y2 = int(y1 + (conv_imgpoint[1] - y1) * scaling_debug)
            cv2.line(img, (x1, y1), (x2, y2), (10, 235, 10), thickness=3)

        cv2.imwrite(str(output / (fname.stem + '_estimated_distortion' + fname.suffix)), img)


def convert_coordinates(img_points, matrix, distortion, fisheye):
    """
    Converts the coordinates from distorted to undistorted, coordinates are not centered
    """
    converted_imgpoints=[]
    f = (matrix[0, 0] + matrix[1, 1]) / 2
    cx = matrix[0, 2]
    cy = matrix[1, 2]
    if fisheye:
        k1 = distortion[0][0]
        k2 = distortion[1][0]
        k3 = distortion[2][0]
        k4 = distortion[3][0]
    else:
        k1 = distortion[0][0]
        k2 = distortion[0][1]
        p1 = distortion[0][2]
        p2 = distortion[0][3]
        k3 = distortion[0][4]

    for points in img_points:
        converted_points = []
        for point in points:
            x = (point[0] - cx) / f
            y = (point[1] - cy) / f
            r = math.sqrt(x ** 2 + y ** 2)
            if fisheye:
                th = math.atan(r)
                th_d = th * (1 - k1 * th ** 2 - k2 * th ** 4 - k3 * th ** 6 - k4 * th ** 8)
                x_ = (r / th_d) * x
                y_ = (r / th_d) * y

            else:
                x_ = x * (1 - k1 * r ** 2 - k2 * r ** 4 - k3 * r ** 6) - p2 * (r ** 2 + 2 * x ** 2) - p1 * 2 * x * y
                y_ = y * (1 - k1 * r ** 2 - k2 * r ** 4 - k3 * r ** 6) - p1 * (r ** 2 + 2 * y ** 2) - p2 * 2 * x * y

            u = round(x_ * f + cx)
            v = round(y_ * f + cy)
            converted_points.append([u, v])
        converted_imgpoints.append(converted_points)

    return converted_imgpoints


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


def detect_calibration_pattern_in_image(file_path, output_folder, debug, 
        kernel_size, distance_scale, relative_threshold_level, distance_threshold):
    """
    Returns the coordinates of the detected corners in 3d object points (corresponds to the real world)
    and the corresponding coordinates in the image calibration plane
    """
    if debug:
        print(file_path.name)
    # define detector
    cbcd = ChessBoardCornerDetector()
    cbcd.kernel_size = kernel_size
    cbcd.distance_scale = distance_scale
    cbcd.relative_threshold_level = relative_threshold_level
    cbcd.distance_threshold = distance_threshold
    # find all corners using the detector
    img = cv2.imread(str(file_path))
    assert img is not None, "Failed to load image"
    corners, coverage, statistics = cbcd.detect_chess_board_corners(img, debug, path_to_image=file_path, path_to_output_folder=output_folder)
    objp = []
    imgp = []
    # fill up the vectors with the corners
    for key, val in corners.items():
        for inner_key, inner_val in val.items():
            objp.append(np.array([key, inner_key, 0]))
            imgp.append(inner_val)
    return np.array(objp, dtype=np.float32), np.array(imgp, dtype=np.float32), coverage, statistics


def calibrate_camera(image_size, objpoints, imgpoints, fisheye=False):
    """
    Calibrate camera with the provided object points and image points
    The image is necessary to get the shape of the image to initialize the intrinsic camera matrix
    """
    if fisheye:
        # the fisheye function is way more demanding on the format of the input...
        objpp = []
        imgpp = []
        for obj_point, image_point in zip(objpoints, imgpoints):
            objpp.append(obj_point.reshape(1, -1, 3))
            imgpp.append(image_point.reshape(1, -1, 2))
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        k = np.zeros((3, 3))
        d = np.zeros((4, 1))
        n_ok = len(objpoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64)]*n_ok
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64)]*n_ok
        retval, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpp, imgpp, image_size, k, d, rvecs, tvecs, calibration_flags)
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return mtx, dist


def print_output(mtx, dist, fisheye):
    """
    Write calibration matrix and distortion coefficients to screen
    """
    print("Calibration matrix: ")
    print(mtx)
    if fisheye:
        print("Distortion parameters (k1, k2, k3, k4):")
    else:
        print("Distortion parameters (k1, k2, p1, p2, k3):")
    print(dist)


def write_output(list_input, output, mtx, dist, coverage_images, stats_before, stats_after, min_percentage_coverage, fisheye):
    """
    Write calibration matrix and distortion coefficients to file together with extended info
    """
    d = datetime.datetime.today()
    output_path = output / ('camera_calibration_extended_' + d.strftime('%Y-%m-%d_%H-%M-%S') + '.txt')
    with output_path.open(mode="w") as f:
        f.write("Time of calibration: \n")
        x = datetime.datetime.now()
        f.write("%s\n\n" % x.strftime("%Y-%m-%d %H:%M:%S"))
        f.write("Calibration matrix: \n")
        for line in mtx:
            f.write(str(line) + '\n')
        if fisheye:
            f.write("\nDistortion parameters (k1, k2, k3, k4):\n")
        else:
            f.write("\nDistortion parameters (k1, k2, p1, p2, k3):\n")
        f.write(str(dist))
        f.write("\n\nSummary statistics:\n")
        hor_before, ver_before, hor_after, ver_after = 0.0, 0.0, 0.0, 0.0
        good_images = 0
        f.write("Image".ljust(25) + "Coverage".ljust(12) + "Avg distor bef".ljust(17) + "Avg distor aft\n")
        for num, fname in enumerate(list_input):
            if coverage_images[num] >= min_percentage_coverage:
                avg_before = round(stats_before[num][0][1] + stats_before[num][1][1] / 2, 2)
                avg_after = round(stats_after[num][0][1] + stats_after[num][1][1] / 2, 2)
                hor_before += round(stats_before[num][0][1], 2)
                ver_before += round(stats_before[num][1][1], 2)
                hor_after += round(stats_after[num][0][1], 2)
                ver_after += round(stats_after[num][1][1], 2)
                f.write(str(fname.name).ljust(25) + (str(coverage_images[num]).rjust(3) + '%').ljust(12) + str(avg_before).ljust(17) + str(avg_after) + '\n')
                good_images += 1
            else:
                f.write(str(fname.name).ljust(25) + (str(coverage_images[num]).rjust(3) + '%').ljust(12) + " -----EXCLUDED-----\n")
        avg_hor_before = round(hor_before / good_images, 2)
        avg_ver_before = round(ver_before / good_images, 2)
        avg_hor_after = round(hor_after / good_images, 2)
        avg_ver_after = round(ver_after / good_images, 2)
        f.write("\nAverage horizontal distortion before:%6.3f pixels from ideal line" % avg_hor_before)
        f.write("\nAverage vertical distortion before:  %6.3f pixels from ideal line" % avg_ver_before)
        f.write("\nAverage horizontal distortion after: %6.3f pixels from ideal line" % avg_hor_after)
        f.write("\nAverage vertical distortion after:   %6.3f pixels from ideal line" % avg_ver_after)
        f.write("\nImages with a coverage lower than " + str(min_percentage_coverage) + "% are excluded from the calibration")
        f.write("\n\nExtended statistics:")
        for num, fname in enumerate(list_input):
            f.write("\n\t" + str(fname.name))
            if coverage_images[num] < min_percentage_coverage:
                f.write("\n\tPercentage of image covered with points: " + str(coverage_images[num]) + "%  -> EXCLUDED")
            else:
                f.write("\n\tPercentage of image covered with points: " + str(coverage_images[num]) + "%")
            f.write("\n\t\tBefore undistorting:")
            f.write("\n\t\t\tHorizontal points : " + str(stats_before[num][0][0]))
            f.write("\n\t\t\tAverage horizontal distortion: %.3f" % stats_before[num][0][1])
            f.write("\n\t\t\tVertical points : " + str(stats_before[num][1][0]))
            f.write("\n\t\t\tAverage vertical distortion: %.3f" % stats_before[num][1][1])
            f.write("\n\t\tAfter undistorting:")
            f.write("\n\t\t\tHorizontal points : " + str(stats_after[num][0][0]))
            f.write("\n\t\t\tAverage horizontal distortion: %.3f" % stats_after[num][0][1])
            f.write("\n\t\t\tVertical points : " + str(stats_after[num][1][0]))
            f.write("\n\t\t\tAverage vertical distortion: %.3f" % stats_after[num][1][1])


def write_calibration(list_input, output, mtx, dist, fisheye):
    """
    Write calibration matrix and distortion coefficients to file
    """
    d = datetime.datetime.today()
    output_path = output / ('calibration_' + d.strftime('%Y-%m-%d_%H-%M-%S') + '.txt')
    with output_path.open(mode="w") as f:
        f.write('-----calibration-----do not edit-----\n')
        f.write("Matrix parameters (fx fy cx cy):\n")
        f.write(str(mtx[0, 0]) + ' ' + str(mtx[1, 1]) + ' ' + str(mtx[0, 2]) + ' ' + str(mtx[1, 2]))
        if fisheye:
            f.write("\n\nDistortion parameters (k1 k2 k3 k4):\n")
            f.write(str(dist[0][0]) + ' ' + str(dist[1][0]) + ' ' + str(dist[2][0]) + ' ' + str(dist[3][0]))
        else:
            f.write("\n\nDistortion parameters (k1 k2 p1 p2 k3):\n")
            f.write(str(dist[0][0]) + ' ' + str(dist[0][1]) + ' ' + str(dist[0][2]) + ' ' + str(dist[0][3]) + ' ' + str(dist[0][4]))
        f.write('\n\n-----info-----may be edited-----\n')
        image_info = open(str(list_input[0]), 'rb')
        tags = exifread.process_file(image_info, details=False)
        try:
            f.write('Camera Model: ' + str(tags['Image Model']))
        except:
            pass
        f.write('\nDistance to calibration: ')


def undistort_images(list_input, output, mtx, dist, fisheye, debug):
    """
    Undistorts all images in the input folder and places them in the output folder
    """
    stats_after = []
    if fisheye:
        for fname in list_input:
            # read image
            img = cv2.imread(str(fname))
            # undistort images
            h,  w = img.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (w, h), cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), undistorted_img)
            cbcd = ChessBoardCornerDetector()
            # make stats
            statistics = cbcd.make_statistics(undistorted_img, debug, output, fname)
            stats_after.append(statistics)
    else:
        for fname in list_input:
            # read image
            img = cv2.imread(str(fname))
            # undistort images, but keep the same camera matrix
            dst = cv2.undistort(img, mtx, dist, None, mtx)
            try: 
                cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), dst)
                cbcd = ChessBoardCornerDetector()
                # make stats
                statistics = cbcd.make_statistics(dst, debug, output, fname)
                stats_after.append(statistics)
            except Exception as e:
                ic(Exception)
    return stats_after


if __name__ == '__main__':
    main()
