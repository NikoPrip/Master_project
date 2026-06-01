"""
VideoGnssCapture.py

Captures video (GStreamer UDP/RTP) and GNSS data (iCA202 NMEA TCP) with GPS UTC
time alignment for precise post-processing synchronization.

GNSS data comes directly from iCA202-3730117 via NMEA TCP — no MQTT needed.
Both the CSV and a metadata file store GPS UTC timestamps, allowing video frames
to be aligned with iCA202-3730072 GNSS data (from gnss_logger.py) by UTC.

Usage:
    python VideoGnssCapture.py --nmea-host 192.168.150.1 --no-preview

Post-processing alignment:
    - video second T  →  GPS UTC = pipeline_gps_utc_s + T  (from .meta.txt)
    - match against gnss_logger.py CSV by gps_utc_s column

Dependencies:
    pip install pyproj
    apt install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-bad \
                gstreamer1.0-plugins-good gstreamer1.0-libav
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from pyproj import Transformer
import csv
import datetime
import math
import socket
import threading
import time
import argparse
from pathlib import Path


_OUTPUT_DIR  = Path(__file__).parent / "gnss_video_pair"
_CSV_DIR     = Path(__file__).parent / "final_test" / "csv_files"
_VIDEO_DIR   = Path(__file__).parent / "final_test" / "videos"


# ---------------------------------------------------------------------------
# NMEA helpers
# ---------------------------------------------------------------------------

def nmea_checksum_ok(sentence: str) -> bool:
    try:
        if sentence[0] != '$' or '*' not in sentence:
            return False
        body, chk = sentence[1:].rsplit('*', 1)
        expected = 0
        for c in body:
            expected ^= ord(c)
        return expected == int(chk.strip(), 16)
    except Exception:
        return False


def nmea_latlon(value: str, hemi: str) -> float:
    if not value:
        return math.nan
    raw = float(value)
    degrees = int(raw / 100)
    minutes = raw - degrees * 100
    dd = degrees + minutes / 60.0
    if hemi in ('S', 'W'):
        dd = -dd
    return dd


def nmea_utc_to_seconds(time_str: str) -> float:
    """Convert NMEA HHMMSS.ss string to seconds since midnight."""
    if not time_str or len(time_str) < 6:
        return math.nan
    try:
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = float(time_str[4:])
        return hh * 3600 + mm * 60 + ss
    except ValueError:
        return math.nan


def transform_to_utm(lat_deg: float, lon_deg: float):
    zone = int((lon_deg + 180) / 6) + 1
    hemisphere = "N" if lat_deg >= 0 else "S"
    epsg = 32600 + zone if hemisphere == "N" else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(lon_deg, lat_deg)
    return f"{zone}{hemisphere}", easting, northing


# ---------------------------------------------------------------------------
# VideoGnssCapture
# ---------------------------------------------------------------------------

class VideoGnssCapture:
    UDP_PORT    = 58004
    CLOCK_RATE  = 90000
    RTP_PAYLOAD = 96
    NMEA_PORT   = 5001

    CSV_COLUMNS = [
        'wall_time',        # ISO wall clock when GNSS sentence arrived
        'video_time_s',     # seconds from pipeline start (wall clock)
        'gps_utc',          # HHMMSS.ss from GGA
        'gps_date',         # DDMMYY from RMC
        'gps_utc_s',        # GPS UTC as seconds since midnight
        'utm_zone',
        'easting_m',
        'northing_m',
        'altitude_m',
        'fix_quality',      # 4=RTK fixed, 5=RTK float
        'num_satellites',
        'hdop',
        'heading_deg',      # from GNHDT dual-antenna baseline
        'speed_knots',
        'ant2_latitude_deg',
        'ant2_longitude_deg',
        'ant2_altitude_m',
    ]

    def __init__(self, output_video: str, gnss_log: str,
                 preview: bool, nmea_host: str, bind_addr: str = None):
        Gst.init(None)

        self.output_video = output_video
        self.gnss_log     = gnss_log
        self.preview      = preview
        self.nmea_host    = nmea_host
        self.bind_addr    = bind_addr

        self._running    = False
        self._pipeline   = None
        self._wall_start = None   # wall clock when pipeline went PLAYING

        self._csv_file   = None
        self._csv_writer = None
        self._gnss_count = 0

        # GPS UTC calibration: gps_utc_s - wall_clock at time of calibration
        # Allows computing GPS UTC for any wall clock timestamp
        self._gps_utc_wall_offset = None

        # GNSS state — updated by NMEA thread, read when writing CSV rows
        self._gnss_lock  = threading.Lock()
        self._gnss_state = {
            'gps_utc': '', 'gps_date': '', 'gps_utc_s': math.nan,
            'latitude_deg': math.nan, 'longitude_deg': math.nan,
            'altitude_m': math.nan, 'fix_quality': -1,
            'num_satellites': 0, 'hdop': math.nan,
            'heading_deg': math.nan, 'speed_knots': math.nan,
            'ant2_latitude_deg': math.nan, 'ant2_longitude_deg': math.nan,
            'ant2_altitude_m': math.nan,
        }

        self._nmea_stop   = threading.Event()
        self._nmea_thread = None
        self._loop        = GLib.MainLoop()

    # ------------------------------------------------------------------
    # NMEA reader (background thread)
    # ------------------------------------------------------------------

    def _nmea_reader_thread(self):
        print(f"[NMEA] Connecting to {self.nmea_host}:{self.NMEA_PORT} ...")
        while not self._nmea_stop.is_set():
            try:
                sock = socket.create_connection(
                    (self.nmea_host, self.NMEA_PORT), timeout=10,
                    source_address=(self.bind_addr, 0) if self.bind_addr else None)
                print(f"[NMEA] Connected to {self.nmea_host}.")
                buf = ""
                while not self._nmea_stop.is_set():
                    chunk = sock.recv(4096).decode('ascii', errors='replace')
                    if not chunk:
                        print("[NMEA] Connection closed by device.")
                        break
                    buf += chunk
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if not line or not nmea_checksum_ok(line):
                            continue
                        self._parse_nmea_line(line)
            except Exception as e:
                if not self._nmea_stop.is_set():
                    print(f"[NMEA] Error: {e} — retrying in 5s ...")
                    time.sleep(5)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    def _parse_nmea_line(self, line: str):
        wall_now = time.time()
        body     = line[1:].split('*')[0]
        fields   = body.split(',')
        sid      = fields[0]

        updates = None
        if sid in ('GNGGA', 'GPGGA'):
            updates = self._parse_gga(fields, wall_now)
        elif sid in ('GNRMC', 'GPRMC'):
            updates = self._parse_rmc(fields)
        elif sid in ('GNHDT', 'GPHDT'):
            updates = self._parse_hdt(fields)
        elif sid == 'PLEIR':
            updates = self._parse_pleir_orp(fields)

        if updates:
            with self._gnss_lock:
                self._gnss_state.update(updates)

        # Write one CSV row per GGA epoch (1 Hz)
        if sid in ('GNGGA', 'GPGGA') and self._running:
            self._write_gnss_row(wall_now)

    def _parse_gga(self, fields, wall_now):
        try:
            utc_str = fields[1]
            utc_s   = nmea_utc_to_seconds(utc_str)

            # Calibrate GPS UTC ↔ wall clock on first valid fix
            if not math.isnan(utc_s) and self._gps_utc_wall_offset is None:
                self._gps_utc_wall_offset = utc_s - wall_now
                print(f"[NMEA] GPS UTC calibrated — offset = "
                      f"{self._gps_utc_wall_offset:.3f}s")

            return {
                'gps_utc':        utc_str,
                'gps_utc_s':      utc_s,
                'latitude_deg':   nmea_latlon(fields[2], fields[3]),
                'longitude_deg':  nmea_latlon(fields[4], fields[5]),
                'fix_quality':    int(fields[6]) if fields[6] else -1,
                'num_satellites': int(fields[7]) if fields[7] else 0,
                'hdop':           float(fields[8]) if fields[8] else math.nan,
                'altitude_m':     float(fields[9]) if fields[9] else math.nan,
            }
        except (IndexError, ValueError):
            return None

    def _parse_rmc(self, fields):
        try:
            return {
                'gps_date':    fields[9] if len(fields) > 9 else '',
                'speed_knots': float(fields[7]) if fields[7] else math.nan,
            }
        except (IndexError, ValueError):
            return None

    def _parse_hdt(self, fields):
        try:
            return {
                'heading_deg': float(fields[1]) if fields[1] else math.nan,
            }
        except (IndexError, ValueError):
            return None

    def _parse_pleir_orp(self, fields):
        if len(fields) < 27 or fields[1] != 'ORP':
            return None
        try:
            return {
                'ant2_latitude_deg':  nmea_latlon(fields[24], 'N'),
                'ant2_longitude_deg': nmea_latlon(fields[25], 'E'),
                'ant2_altitude_m':    float(fields[26]) if fields[26] else math.nan,
            }
        except (IndexError, ValueError):
            return None

    def _write_gnss_row(self, wall_now: float):
        video_time_s = (wall_now - self._wall_start) \
                       if self._wall_start is not None else 0.0
        iso_time = datetime.datetime.fromtimestamp(wall_now).isoformat(
            timespec='milliseconds')

        with self._gnss_lock:
            state = dict(self._gnss_state)

        lat = state['latitude_deg']
        lon = state['longitude_deg']
        try:
            utm_zone, easting, northing = transform_to_utm(lat, lon)
        except Exception:
            utm_zone, easting, northing = 'N/A', math.nan, math.nan

        def fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if not math.isnan(v) else ''

        row = {
            'wall_time':      iso_time,
            'video_time_s':   f"{video_time_s:.3f}",
            'gps_utc':        state['gps_utc'],
            'gps_date':       state['gps_date'],
            'gps_utc_s':      fmt(state['gps_utc_s'], 2),
            'utm_zone':       utm_zone,
            'easting_m':      fmt(easting, 3),
            'northing_m':     fmt(northing, 3),
            'altitude_m':     fmt(state['altitude_m'], 3),
            'fix_quality':    state['fix_quality'],
            'num_satellites': state['num_satellites'],
            'hdop':           fmt(state['hdop'], 2),
            'heading_deg':    fmt(state['heading_deg'], 4),
            'speed_knots':    fmt(state['speed_knots'], 4),
            'ant2_latitude_deg':  fmt(state['ant2_latitude_deg'], 7),
            'ant2_longitude_deg': fmt(state['ant2_longitude_deg'], 7),
            'ant2_altitude_m':    fmt(state['ant2_altitude_m'], 3),
        }

        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        self._gnss_count += 1
        fix_label = {4: 'RTK-fixed', 5: 'RTK-float'}.get(
            state['fix_quality'], f"fix={state['fix_quality']}")
        print(f"[{video_time_s:8.3f}s] GNSS #{self._gnss_count}: "
              f"UTC={state['gps_utc']}  "
              f"E={easting:.1f}  N={northing:.1f}  "
              f"alt={state['altitude_m']:.1f}m  "
              f"hdg={fmt(state['heading_deg'], 2)}°  {fix_label}")

    # ------------------------------------------------------------------
    # GStreamer pipeline
    # ------------------------------------------------------------------

    def _build_pipeline_str(self) -> str:
        sink = ("avdec_h264 ! autovideosink sync=true"
                if self.preview else "fakesink sync=false")
        return (
            f"udpsrc port={self.UDP_PORT} buffer-size=212992 do-timestamp=true "
            f"! application/x-rtp,media=video,clock-rate={self.CLOCK_RATE},"
            f"payload={self.RTP_PAYLOAD},encoding-name=H264 "
            f"! rtpjitterbuffer latency=50 "
            f"! rtph264depay "
            f"! h264parse "
            f"! tee name=t "
            f"  t. ! queue ! mp4mux ! filesink location={self.output_video} "
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
                    print(f"[GST] Pipeline PLAYING — "
                          f"wall_start={self._wall_start:.3f}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> bool:
        Path(self.output_video).parent.mkdir(parents=True, exist_ok=True)

        self._csv_file = open(self.gnss_log, 'w', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self.CSV_COLUMNS)
        self._csv_writer.writeheader()

        # Start NMEA thread
        self._nmea_stop.clear()
        self._nmea_thread = threading.Thread(
            target=self._nmea_reader_thread, daemon=True)
        self._nmea_thread.start()

        # Build and start GStreamer pipeline
        pipeline_str = self._build_pipeline_str()
        print(f"[GST] Pipeline:\n  {pipeline_str}\n")
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._pipeline.set_latency(100 * Gst.MSECOND)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[GST] Failed to set pipeline to PLAYING.")
            return False

        self._running = True
        print(f"[INFO] Saving video → {self.output_video}")
        print(f"[INFO] Saving GNSS  → {self.gnss_log}")
        print(f"[INFO] NMEA source  → {self.nmea_host}:{self.NMEA_PORT}")
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
        self._nmea_stop.set()

        if self._pipeline:
            self._pipeline.send_event(Gst.Event.new_eos())
            time.sleep(1.0)
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        if self._csv_file:
            self._csv_file.close()

        # Write metadata file for post-processing alignment
        meta_path = Path(self.gnss_log).with_suffix('.meta.txt')
        with open(meta_path, 'w') as f:
            f.write(f"video_file={self.output_video}\n")
            f.write(f"gnss_log={self.gnss_log}\n")
            f.write(f"nmea_host={self.nmea_host}\n")
            if self._wall_start is not None:
                f.write(f"pipeline_wall_start={self._wall_start:.6f}\n")
            if self._gps_utc_wall_offset is not None:
                f.write(f"gps_utc_wall_offset_s={self._gps_utc_wall_offset:.6f}\n")
            if self._wall_start is not None and self._gps_utc_wall_offset is not None:
                gps_utc_at_start = self._wall_start + self._gps_utc_wall_offset
                f.write(f"pipeline_gps_utc_s={gps_utc_at_start:.3f}\n")
                f.write(
                    f"# video frame at T seconds → GPS UTC = "
                    f"pipeline_gps_utc_s + T\n"
                )
        print(f"[INFO] Metadata     → {meta_path}")
        print(f"[INFO] Done. Logged {self._gnss_count} GNSS entries.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Synchronised video + GNSS capture via NMEA TCP")
    p.add_argument("--name",         default=None,
                   help="Base name for output files, e.g. 'aruco_1' → "
                        "videos/aruco_1.mp4 and csv_files/gnss_log_3730117_aruco_1.csv. "
                        "Overrides --output-video and --gnss-log defaults.")
    p.add_argument("--output-video", default=None,
                   help="Output MP4 file (default: videos/<name>.mp4 or videos/output.mp4)")
    p.add_argument("--gnss-log",     default=None,
                   help="Output GNSS CSV log (default: csv_files/gnss_log_3730117_<name>.csv)")
    p.add_argument("--no-preview",   action="store_true",
                   help="Disable live video preview")
    p.add_argument("--nmea-host",    default="192.168.150.1",
                   help="IP of iCA202-3730117 NMEA stream (default: 192.168.150.1)")
    p.add_argument("--bind-addr",    default=None,
                   help="Local IP to bind to (use when two iCA202s are connected)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    name = args.name or "output"
    output_video = args.output_video or str(_VIDEO_DIR   / f"{name}.mp4")
    gnss_log     = args.gnss_log     or str(_CSV_DIR     / f"gnss_log_3730117_{name}.csv")

    VideoGnssCapture(
        output_video=output_video,
        gnss_log=gnss_log,
        preview=not args.no_preview,
        nmea_host=args.nmea_host,
        bind_addr=args.bind_addr,
    ).run()
