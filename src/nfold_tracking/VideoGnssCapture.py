"""
VideoGnssCapture.py

Simultaneously captures video (via GStreamer UDP/RTP pipeline) and GNSS data
(via MQTT), logging GNSS entries with timestamps relative to the video timeline.
GNSS coordinates are stored in UTM (Universal Transverse Mercator) format.

Dependencies:
    pip install paho-mqtt pyproj
    apt install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-bad \
                gstreamer1.0-plugins-good gstreamer1.0-libav
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import paho.mqtt.client as mqtt
from pyproj import Transformer
import json
import math
import csv
import time
import datetime
import os
import argparse
from pathlib import Path


# Output directory relative to this script
_OUTPUT_DIR = Path(__file__).parent / "gnss_video_pair"


def transform_to_utm(lat_rad: float, lon_rad: float):
    """Convert WGS84 radians to UTM easting/northing.

    Returns:
        (zone_str, easting_m, northing_m)  e.g. ("32N", 587423.1, 6219345.7)
    """
    lat = math.degrees(lat_rad)
    lon = math.degrees(lon_rad)

    zone = int((lon + 180) / 6) + 1
    hemisphere = "N" if lat >= 0 else "S"
    epsg = 32600 + zone if hemisphere == "N" else 32700 + zone

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return f"{zone}{hemisphere}", easting, northing


class VideoGnssCapture:
    MQTT_HOST  = "192.168.40.2"
    MQTT_PORT  = 1883
    MQTT_TOPIC = "GNSSInterface/Data"

    UDP_PORT      = 58004
    CLOCK_RATE    = 90000
    RTP_PAYLOAD   = 96

    def __init__(self, output_video: str = str(_OUTPUT_DIR / "output.mp4"),
                 gnss_log: str = str(_OUTPUT_DIR / "gnss_log.csv"),
                 preview: bool = True):
        Gst.init(None)

        self.output_video = output_video
        self.gnss_log     = gnss_log
        self.preview      = preview

        self._running           = False
        self._pipeline          = None
        self._pipeline_start_ns = None   # GStreamer clock time when PLAYING began
        self._wall_start        = None   # system wall time at that moment

        self._csv_file   = None
        self._csv_writer = None
        self._gnss_count = 0

        self._loop = GLib.MainLoop()

        # MQTT
        self._mqtt = mqtt.Client()
        self._mqtt.on_connect = self._on_mqtt_connect
        self._mqtt.on_message = self._on_mqtt_message

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Connected to {self.MQTT_HOST}:{self.MQTT_PORT}")
            client.subscribe(self.MQTT_TOPIC)
            print(f"[MQTT] Subscribed to {self.MQTT_TOPIC}")
        else:
            print(f"[MQTT] Connection failed (rc={rc})")

    def _on_mqtt_message(self, client, userdata, msg):
        if not self._running:
            return

        wall_now = time.time()
        video_ts_s = wall_now - self._wall_start if self._wall_start is not None else 0.0
        iso_time = datetime.datetime.fromtimestamp(wall_now).isoformat(timespec="milliseconds")

        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception as e:
            print(f"[GNSS] Could not parse payload: {e}")
            return

        lat      = float(data.get("LAT",        0))
        lon      = float(data.get("LONG",       0))
        alt      = float(data.get("ALT",        0))
        speed    = float(data.get("SPEED",      0))
        heading  = float(data.get("HEADING",    0))
        acc_h    = float(data.get("ACCURACY_H", 0))
        acc_v    = float(data.get("ACCURACY_V", 0))
        has_fix  = data.get("HAS_FIX",    "0")
        trusted  = data.get("IS_TRUSTED", "0")
        gnss_ts  = data.get("timestamp",  "")

        try:
            utm_zone, easting, northing = transform_to_utm(lat, lon)
        except Exception as e:
            print(f"[GNSS] UTM conversion failed: {e}")
            utm_zone, easting, northing = "N/A", float("nan"), float("nan")

        row = {
            "system_time":  iso_time,
            "video_time_s": f"{video_ts_s:.3f}",
            "gnss_time":    gnss_ts,
            "utm_zone":     utm_zone,
            "easting_m":    f"{easting:.3f}",
            "northing_m":   f"{northing:.3f}",
            "altitude_m":   f"{alt:.3f}",
            "speed":        f"{speed:.4f}",
            "heading":      f"{heading:.4f}",
            "accuracy_h_m": f"{acc_h:.3f}",
            "accuracy_v_m": f"{acc_v:.3f}",
            "has_fix":      has_fix,
            "is_trusted":   trusted,
        }

        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        self._gnss_count += 1
        print(f"[{video_ts_s:8.3f}s] GNSS #{self._gnss_count}: "
              f"zone={utm_zone}  E={easting:.1f}  N={northing:.1f}  "
              f"alt={alt:.1f}m  fix={has_fix}")

    # ------------------------------------------------------------------
    # GStreamer pipeline
    # ------------------------------------------------------------------

    def _build_pipeline_str(self) -> str:
        sink = (
            "avdec_h264 ! xvimagesink sync=true"
            if self.preview
            else "avdec_h264 ! fakesink"
        )
        return (
            f"udpsrc port={self.UDP_PORT} buffer-size=212992 do-timestamp=true "
            f"! application/x-rtp,media=video,clock-rate={self.CLOCK_RATE},"
            f"payload={self.RTP_PAYLOAD},encoding-name=H264 "
            f"! rtph264depay "
            f"! h264parse "
            f"! tee name=t "
            f"  t. ! queue ! qtmux fragment-duration=1000 "
            f"       ! filesink location={self.output_video} "
            f"  t. ! queue ! {sink}"
        )

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[GST] End of stream.")
            self._loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[GST] Error: {err.message}")
            if debug:
                print(f"[GST] Debug: {debug}")
            self._loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                _, new, _ = message.parse_state_changed()
                if new == Gst.State.PLAYING and self._wall_start is None:
                    self._wall_start = time.time()
                    clock = self._pipeline.get_clock()
                    if clock:
                        self._pipeline_start_ns = clock.get_time()
                    print("[GST] Pipeline is PLAYING — timestamps active.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> bool:
        # Ensure output directory exists
        Path(self.output_video).parent.mkdir(parents=True, exist_ok=True)

        # Open CSV log
        self._csv_file = open(self.gnss_log, "w", newline="")
        fieldnames = [
            "system_time", "video_time_s", "gnss_time",
            "utm_zone", "easting_m", "northing_m", "altitude_m",
            "speed", "heading", "accuracy_h_m", "accuracy_v_m",
            "has_fix", "is_trusted",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

        # Start MQTT (non-blocking loop)
        try:
            self._mqtt.connect(self.MQTT_HOST, self.MQTT_PORT, keepalive=60)
        except Exception as e:
            print(f"[MQTT] Could not connect: {e}")
            return False
        self._mqtt.loop_start()

        # Build GStreamer pipeline
        pipeline_str = self._build_pipeline_str()
        print(f"[GST] Pipeline:\n  {pipeline_str}\n")
        self._pipeline = Gst.parse_launch(pipeline_str)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[GST] Failed to set pipeline to PLAYING.")
            return False

        self._running = True
        print(f"[INFO] Saving video  → {self.output_video}")
        print(f"[INFO] Saving GNSS   → {self.gnss_log}")
        print("[INFO] Press Ctrl-C to stop.\n")
        return True

    def run(self):
        if not self.start():
            return
        try:
            self._loop.run()
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        self._running = False

        if self._pipeline:
            self._pipeline.send_event(Gst.Event.new_eos())
            # Give EOS a moment to flush file
            time.sleep(1.0)
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        self._mqtt.loop_stop()
        self._mqtt.disconnect()

        if self._csv_file:
            self._csv_file.close()

        print(f"[INFO] Done. Logged {self._gnss_count} GNSS entries.")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Synchronised video + GNSS capture")
    p.add_argument("--output-video", default=str(_OUTPUT_DIR / "output.mp4"),
                   help="Output MP4 file (default: gnss_video_pair/output.mp4)")
    p.add_argument("--gnss-log", default=str(_OUTPUT_DIR / "gnss_log.csv"),
                   help="Output GNSS CSV log (default: gnss_video_pair/gnss_log.csv)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable live video preview window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture = VideoGnssCapture(
        output_video=args.output_video,
        gnss_log=args.gnss_log,
        preview=not args.no_preview,
    )
    capture.run()
