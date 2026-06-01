# -*- coding: utf-8 -*-
"""
Marker tracker for locating n-fold edges in images using convolution.

@author: Henrik Skov Midtiby
"""
import cv2
import numpy as np
import math
from MarkerPose import MarkerPose
import time
from itertools import permutations


class MarkerTracker:
    """
    Purpose: Locate a certain marker in an image.
    """

    def __init__(self, order, kernel_size, scale_factor):
        self.kernel_size = kernel_size
        (kernel_real, kernel_imag) = self.generate_symmetry_detector_kernel(order, kernel_size)

        self.order = order
        self.mat_real = kernel_real / scale_factor
        self.mat_imag = kernel_imag / scale_factor

        self.frame_real = None
        self.frame_imag = None
        self.last_marker_location = None
        self.orientation = None
        self.track_marker_with_missing_black_leg = True

        #her
        self.expected_ratios = None

        # Create kernel used to remove arm in quality-measure
        (kernel_remove_arm_real, kernel_remove_arm_imag) = self.generate_symmetry_detector_kernel(1, self.kernel_size)
        self.kernelComplex = np.array(kernel_real + 1j*kernel_imag, dtype=complex)
        self.KernelRemoveArmComplex = np.array(kernel_remove_arm_real + 1j*kernel_remove_arm_imag, dtype=complex)

        # Values used in quality-measure
        absolute = np.absolute(self.kernelComplex)
        self.threshold = 0.4*absolute.max()
        self.quality = None
        self.y1 = int(math.floor(float(self.kernel_size)/2))
        self.y2 = int(math.ceil(float(self.kernel_size)/2))
        self.x1 = int(math.floor(float(self.kernel_size)/2))
        self.x2 = int(math.ceil(float(self.kernel_size)/2))

        # Information about the located marker.
        self.pose = None


    @staticmethod
    def generate_symmetry_detector_kernel(order, kernel_size):
        # type: (int, int) -> numpy.ndarray
        value_range = np.linspace(-1, 1, kernel_size)
        temp1 = np.meshgrid(value_range, value_range)
        kernel = temp1[0] + 1j * temp1[1]

        magnitude = abs(kernel)
        kernel = np.power(kernel, order)
        kernel = kernel * np.exp(-8 * magnitude ** 2)

        return np.real(kernel), np.imag(kernel)

    def refine_marker_location(self):
        try: 
            delta = 1
            # Fit a parabola to the frame_sum_squared marker response
            # and then locate the top of the parabola.
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.frame_sum_squared)
            x = max_loc[1]
            y = max_loc[0]
            frame_sum_squared_cutout = self.frame_sum_squared[x-delta:x+delta+1, y-delta:y+delta+1]
            # Taking the square root of the frame_sum_squared improves the accuracy of the 
            # refied marker position.
            frame_sum_squared_cutout = np.sqrt(frame_sum_squared_cutout)

            nx, ny = (1 + 2*delta, 1 + 2*delta)
            x = np.linspace(-delta, delta, nx)
            y = np.linspace(-delta, delta, ny)
            xv, yv = np.meshgrid(x, y)

            xv = xv.ravel()
            yv = yv.ravel()

            coefficients = np.concatenate([[xv**2], [xv], [yv**2], [yv], [yv**0]], axis = 0).transpose()
            values = frame_sum_squared_cutout.ravel().reshape(-1, 1)
            solution, residuals, rank, s = np.linalg.lstsq(coefficients, values, rcond=None)
            dx = -solution[1] / (2*solution[0])
            dy = -solution[3] / (2*solution[2])
            return dx[0], dy[0]
        except np.linalg.LinAlgError:
            # This error is triggered when the marker is detected close to an edge.
            # In that case the refine method bails out and returns two zeros.
            return 0, 0

    def locate_marker_init(self, frame):
        assert len(frame.shape) == 2, "Input image is not a single channel image."
        self.frame_real = frame.copy()
        self.frame_imag = frame.copy()

        # Calculate convolution and determine response strength.
        self.frame_real = cv2.filter2D(self.frame_real, cv2.CV_32F, self.mat_real)
        self.frame_imag = cv2.filter2D(self.frame_imag, cv2.CV_32F, self.mat_imag)
        frame_real_squared = cv2.multiply(self.frame_real, self.frame_real, dtype=cv2.CV_32F)
        frame_imag_squared = cv2.multiply(self.frame_imag, self.frame_imag, dtype=cv2.CV_32F)
        self.frame_sum_squared = cv2.add(frame_real_squared, frame_imag_squared, dtype=cv2.CV_32F)

    def locate_marker(self, frame):
        #det her
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.frame_sum_squared)
        self.last_marker_location = max_loc
        self.determine_marker_orientation(frame)
        self.determine_marker_quality(frame)
        dx, dy = self.refine_marker_location()
        #print(f"dx: {dx: 0.2f}  dy: {dy: 0.2f}")
        max_loc = (max_loc[0] + dx, max_loc[1] + dy)

        self.pose = MarkerPose(max_loc[0], max_loc[1], self.orientation, self.quality, self.order)
        return self.pose

    def determine_marker_orientation(self, frame):
        (xm, ym) = self.last_marker_location
        real_value = self.frame_real[ym, xm]
        imag_value = self.frame_imag[ym, xm]
        self.orientation = (math.atan2(-real_value, imag_value) - math.pi / 2) / self.order

        max_value = 0
        max_orientation = self.orientation
        search_distance = self.kernel_size / 3
        for k in range(self.order):
            orient = self.orientation + 2 * k * math.pi / self.order
            xm2 = int(xm + search_distance * math.cos(orient))
            ym2 = int(ym + search_distance * math.sin(orient))
            try:
                intensity = frame[ym2, xm2]
                if intensity > max_value:
                    max_value = intensity
                    max_orientation = orient
            except Exception as e:
                print("determineMarkerOrientation: error: %d %d %d %d" % (ym2, xm2, frame.shape[1], frame.shape[0]))
                print(e)
                pass

        self.orientation = self.limit_angle_to_range(max_orientation)

    @staticmethod
    def limit_angle_to_range(angle):
        while angle < math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle

    def determine_marker_quality(self, frame):
        (bright_regions, dark_regions) = self.generate_template_for_quality_estimator()
        # cv2.imshow("bright_regions", 255*bright_regions)
        # cv2.imshow("dark_regions", 255*dark_regions)

        try:
            frame_img = self.extract_window_around_maker_location(frame)
            (bright_mean, bright_std) = cv2.meanStdDev(frame_img, mask=bright_regions)
            (dark_mean, dark_std) = cv2.meanStdDev(frame_img, mask=dark_regions)

            mean_difference = bright_mean - dark_mean
            normalised_mean_difference = mean_difference / (0.5*bright_std + 0.5*dark_std)
            # Ugly hack for translating the normalised_mean_differences to the range [0, 1]
            temp_value_for_quality = 1 - 1/(1 + math.exp(0.75*(-7+normalised_mean_difference)))
            self.quality = temp_value_for_quality
        except Exception as e:
            print("error")
            print(e)
            self.quality = 0.0
            return

    def extract_window_around_maker_location(self, frame):
        (xm, ym) = self.last_marker_location
        frame_tmp = np.array(frame[ym - self.y1:ym + self.y2, xm - self.x1:xm + self.x2])
        frame_img = frame_tmp.astype(np.uint8)
        return frame_img

    def generate_template_for_quality_estimator(self):
        phase = np.exp((self.limit_angle_to_range(-self.orientation)) * 1j)
        angle_threshold = 3.14 / (2 * self.order)
        t3 = np.angle(self.KernelRemoveArmComplex * phase) < angle_threshold
        t4 = np.angle(self.KernelRemoveArmComplex * phase) > -angle_threshold

        signed_mask = 1 - 2 * (t3 & t4)
        adjusted_kernel = self.kernelComplex * np.power(phase, self.order)
        if self.track_marker_with_missing_black_leg:
            adjusted_kernel *= signed_mask
        bright_regions = (adjusted_kernel.real < -self.threshold).astype(np.uint8)
        dark_regions = (adjusted_kernel.real > self.threshold).astype(np.uint8)

        return bright_regions, dark_regions   

    #Casper Code

    def detect_multiple_markers(self, frame):
        poses = []
        reference_intensity = None
        while True:
            marker = self.locate_marker(frame)
            if marker.x < 0 or marker.y < 0:
                break
            marker_intensity = self.frame_sum_squared[int(marker.y), int(marker.x)]
            if reference_intensity is None:
                reference_intensity = marker_intensity
            noise = 1000
            if marker_intensity / (reference_intensity + noise) <= 0.05:
                break
            poses.append(marker)
            radius = 3
            for y in range(max(0, int(marker.y) - radius), min(self.frame_sum_squared.shape[0], int(marker.y) + radius)):
                for x in range(max(0, int(marker.x) - radius), min(self.frame_sum_squared.shape[1], int(marker.x) + radius)):
                    self.frame_sum_squared[y, x] = 0
        number_of_markers = len(poses)

        return poses, number_of_markers, reference_intensity  

    def distances_between_markers(self,poses,number_of_markers):
        distances_between_markers = [[] for _ in range(number_of_markers)]
        for i in range(number_of_markers):
            for j in range(number_of_markers):
                if i != j:
                    distances_between_markers[i].append(np.sqrt((poses[i].x - poses[j].x)**2 + (poses[i].y - poses[j].y)**2))
                elif i == j:
                    distances_between_markers[i].append(np.inf)
        return distances_between_markers
    
    def validate_marker_pair(self, current_list, tolerance):
        distance_matrix = self.distances_between_markers(current_list, len(current_list))
        distances = []
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):  # Only consider i < j to avoid duplicates
                distances.append(distance_matrix[i][j])
        base_distance = min(distances)
        normalized_distances = [distance / base_distance for distance in distances]
        if normalized_distances[0] == 1.0:
            if all(abs(nd - er) < tolerance for nd, er in zip(normalized_distances, self.expected_ratios)):
                return True
        return False

    def detect_marker_pairs(self,poses,distances_between_markers):
        marker_pairs = []
        distances_between_markers_copy = distances_between_markers.copy()
        if len(poses) >= 5:
            for pose in poses:
                if not any(pose in pair for pair in marker_pairs):
                    current_list = []
                    current_list.append(pose)
                    for _ in range(4):
                        current_pose_index = poses.index(pose)
                        closest_marker = min(distances_between_markers_copy[current_pose_index])
                        closest_marker_index = distances_between_markers_copy[current_pose_index].index(closest_marker)
                        current_list.append(poses[closest_marker_index])
                        distances_between_markers_copy[current_pose_index][closest_marker_index] = np.inf
                    permutations_list = permutations(current_list[1:])
                    for perm in permutations_list:
                        current_list = [current_list[0]] + list(perm)
                        if self.validate_marker_pair(current_list, tolerance=0.8):
                            marker_pairs.append(current_list)
                            break
        return marker_pairs
    
    def numerate_markers(self,marker_pairs):
        for pairs in marker_pairs:
            for i in range(len(pairs)):
                pairs[i].number = i

    def marker_corners(self, marker_pairs):
        marker_corners = []
        for pairs in marker_pairs:
            corners = []
            for pair in pairs:
                if pair.number != 0:
                    corners.append([pair.x, pair.y])
            corners = np.array(corners, dtype=np.float32).reshape(1, 4, 2) #reshape to match aruco format
            marker_corners.append(corners)
        return marker_corners

