import cv2
import argparse
from pathlib import Path
from icecream import ic


def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('input', type=lambda p: Path(p).absolute(), help='input video file')
    parser.add_argument('output', type=lambda p: Path(p).absolute(), help='output directory')

    args = parser.parse_args()

    print(args.input)
    print(args.output)

    cap = cv2.VideoCapture(str(args.input))
    counter = 0
    frame_counter = 0

    while cap.isOpened():
        ret_val, frame = cap.read()
        if not ret_val:
            fake_frame_skip = 1000
            while not ret_val and fake_frame_skip > 0:
                fake_frame_skip -= 1
                ret_val, frame = cap.read()
            if not ret_val:
                raise Exception(
                    f'Video stream ended unexpectedly. See this issue for details: https://github.com/ultralytics/yolov5/issues/2064')

        frame_counter += 1
        ic(frame_counter)
        if ret_val is False:
            break

        cv2.imshow("frame", frame)
        
        k = cv2.waitKey(30)
        if k == ord('q'):
            break
        if k == ord('s'):
            filename = "%s/frame%03d.jpg" % (args.output, counter)
            counter += 1
            cv2.imwrite(filename, frame)
            print(f"Saved frame { frame_counter } to the file { filename }")


main()
