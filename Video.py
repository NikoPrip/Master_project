import os
import cv2

class FrameIterator():
    def __init__(self, filename, output_dir="Images"):
        self.cap = cv2.VideoCapture(filename)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # make sure folder exists

    def frame_generator(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
        self.cap.release()

    def main(self):
        frame_count = 0
        for frame in self.frame_generator():
            # Show frame (optional)
            #cv2.imshow('frame', frame)

            # Save frame with unique filename
            filename = os.path.join(self.output_dir, f"frame_{frame_count:05d}.jpeg")
            if frame_count % 15 == 0:
                cv2.imwrite(filename, frame)

            frame_count += 1

            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_file = "output.mp4"
    fi = FrameIterator(video_file)
    fi.main()