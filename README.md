# Master_project

# Commands for image and video:
192.168.40.90
gst-launch-1.0 -v -t --gst-debug-level=3 pipeline. \( async-handling=true udpsrc port=58004 timeout=500000000 buffer-size=212992 do-timestamp=true ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! video/x-h264,stream-format=\(string\)byte-stream,alignment=\(string\)au ! queue ! avdec_h264 ! xvimagesink sync=true  \)

Show video and save video:
gst-launch-1.0 -v -t \
  udpsrc port=58004 buffer-size=212992 do-timestamp=true \
  ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 \
  ! rtph264depay \
  ! h264parse \
  ! tee name=t \
    t. ! queue ! qtmux fragment-duration=1000 ! filesink location=output.mp4 \
    t. ! queue ! avdec_h264 ! xvimagesink sync=true


192.168.40.91
gst-launch-1.0 -v -t --gst-debug-level=3 pipeline. \( async-handling=true udpsrc port=58006 timeout=500000000 buffer-size=212992 do-timestamp=true ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! video/x-h264,stream-format=\(string\)byte-stream,alignment=\(string\)au ! queue ! avdec_h264 ! xvimagesink sync=true  \)

Show video and save video:
gst-launch-1.0 -v -t \
  udpsrc port=58006 buffer-size=212992 do-timestamp=true \
  ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 \
  ! rtph264depay \
  ! h264parse \
  ! tee name=t \
    t. ! queue ! qtmux fragment-duration=1000 ! filesink location=output.mp4 \
    t. ! queue ! avdec_h264 ! xvimagesink sync=true


Image:
gst-launch-1.0 -v -t --gst-debug-level=3 pipeline. \( async-handling=true udpsrc port=58004 timeout=500000000 buffer-size=212992 do-timestamp=true ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! video/x-h264,stream-format=\(string\)byte-stream,alignment=\(string\)au ! queue ! avdec_h264 ! jpegenc ! filesink location=test.jpeg  \)