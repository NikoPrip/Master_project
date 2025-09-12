## Calibrate camera and undistort images
This library consists of:
 - two python libraries with functions to identify corners in a checker board
 - a script that uses the corners in test images to calculate the parameters to calibrate a camera
 - a script that uses the parameters to undistort images made with this camera

## Install actions
```
pipenv install
```

## Run the calibration
Enter the working directory (where the scripts are placed), put check board images in the input folder and issue the following commands:
```
pipenv run python calibration.py --input <folder-name> --output <folder-name> --fisheye --debug --min_covarage=25 --scaling_debug=1
```

All parameters are optional:
 - input: if not provided, input folder is considered to be $wd/input
 - output: if not provided, output folder will be created as $wd/output
 - fisheye: set this if the you want to use the fish eye camera model
 - debug: set to get intermediate steps in corner detection and an estimation of the camera distortion
 - min_coverage: minimum percentage of the image that needs to be covered with detected corners in order to be considered during the calibration. Lower if not enough images are considered
 - scaling_debug: scaling used while drawing the estimation of the distortion. Use if the distortion is very small and not visible in the estimation

## Undistort images
Enter the working directory (where the scripts are placed), put distorted images in the input folder and issue the following commands:
```
pipenv run python undistort.py --input <folder-name> --output <folder-name> --calibration <calibration-file>
```

All parameters are optional:
 - input: if not provided, input folder is considered to be $wd/input
 - output: if not provided, output folder will be created as $wd/output
 - calibration: if not provided, script looks for calibration file in input folder
