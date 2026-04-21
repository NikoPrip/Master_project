"""
GNSS logger for Leica iCA202 (iCON aps 200).

Connects to the device's NMEA TCP stream, parses the relevant sentences,
and writes one CSV row per GGA update (1 Hz).

Usage:
    python gnss_logger.py output.csv
    python gnss_logger.py output.csv --host 192.168.151.1 --port 5001

CSV columns:
    timestamp_utc       - HHMMSS.ss from GGA
    date_utc            - DDMMYY from RMC  (empty until first RMC)
    latitude_deg        - decimal degrees, positive = North  (antenna 1)
    longitude_deg       - decimal degrees, positive = East   (antenna 1)
    height_m            - ellipsoidal height above WGS84 (metres, antenna 1)
    fix_quality         - 0=no fix, 1=GPS, 2=DGPS, 4=RTK fixed, 5=RTK float
    num_satellites      - number of satellites in use
    hdop                - horizontal dilution of precision
    heading_deg         - true heading from dual-antenna baseline (GNHDT), NaN if unavailable
    speed_knots         - speed over ground from RMC, NaN if unavailable
    ant2_latitude_deg   - decimal degrees, positive = North  (antenna 2 / heading antenna)
    ant2_longitude_deg  - decimal degrees, positive = East   (antenna 2 / heading antenna)
    ant2_height_m       - ellipsoidal height above WGS84 (metres, antenna 2)
"""

import argparse
import csv
import math
import socket
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# NMEA helpers
# ---------------------------------------------------------------------------

def nmea_checksum_ok(sentence: str) -> bool:
    """Return True if the NMEA checksum is valid."""
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
    """Convert NMEA DDMM.MMMM + hemisphere char to decimal degrees."""
    if not value:
        return math.nan
    raw = float(value)
    degrees = int(raw / 100)
    minutes = raw - degrees * 100
    dd = degrees + minutes / 60.0
    if hemi in ('S', 'W'):
        dd = -dd
    return dd


# ---------------------------------------------------------------------------
# Sentence parsers — each returns a dict of fields to update, or None
# ---------------------------------------------------------------------------

def parse_gga(fields):
    """$GNGGA: time, position, fix quality, satellites, HDOP, height."""
    # fields[0] is the sentence ID, data starts at fields[1]
    try:
        return {
            'timestamp_utc': fields[1],
            'latitude_deg':  nmea_latlon(fields[2], fields[3]),
            'longitude_deg': nmea_latlon(fields[4], fields[5]),
            'fix_quality':   int(fields[6]) if fields[6] else -1,
            'num_satellites': int(fields[7]) if fields[7] else 0,
            'hdop':          float(fields[8]) if fields[8] else math.nan,
            'height_m':      float(fields[9]) if fields[9] else math.nan,
        }
    except (IndexError, ValueError):
        return None


def parse_rmc(fields):
    """$GNRMC: date, speed, course over ground."""
    try:
        return {
            'date_utc':    fields[9] if len(fields) > 9 else '',
            'speed_knots': float(fields[7]) if fields[7] else math.nan,
        }
    except (IndexError, ValueError):
        return None


def parse_hdt(fields):
    """$GNHDT: true heading from dual-antenna baseline."""
    try:
        return {
            'heading_deg': float(fields[1]) if fields[1] else math.nan,
        }
    except (IndexError, ValueError):
        return None


def parse_pleir_orp(fields):
    """$PLEIR,ORP: Leica proprietary dual-antenna position sentence.

    Layout (0-based field indices after splitting on comma):
      0  PLEIR
      1  ORP
      2  fix type
      3  (reserved)
      4  number of antenna solutions
      5-16  antenna 1: time, date, sats, sigE, sigN, sigU, baseline_m, ?, ?, lat, lon, h
      17-26 antenna 2: id, time, date, sats, sigE, sigN, sigU, lat, lon, h
      27-31 heading: time, date, sats, heading_deg, sigma

    Lat/lon are in NMEA DDMM.MMMM format, assumed North/East (positive).
    """
    if len(fields) < 27 or fields[1] != 'ORP':
        return None
    try:
        return {
            'ant2_latitude_deg':  nmea_latlon(fields[24], 'N'),
            'ant2_longitude_deg': nmea_latlon(fields[25], 'E'),
            'ant2_height_m':      float(fields[26]) if fields[26] else math.nan,
        }
    except (IndexError, ValueError):
        return None


PARSERS = {
    'GNGGA': parse_gga,
    'GNRMC': parse_rmc,
    'GNHDT': parse_hdt,
    'PLEIR': parse_pleir_orp,
    # Also accept GP/GL/GA prefix variants
    'GPGGA': parse_gga,
    'GPRMC': parse_rmc,
    'GPHDT': parse_hdt,
}

CSV_COLUMNS = [
    'timestamp_utc', 'date_utc',
    'latitude_deg', 'longitude_deg', 'height_m',
    'fix_quality', 'num_satellites', 'hdop',
    'heading_deg', 'speed_knots',
    'ant2_latitude_deg', 'ant2_longitude_deg', 'ant2_height_m',
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Log GNSS data from iCA202 to CSV.')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--host', default='192.168.151.1', help='Device IP (default: 192.168.151.1 for iCA202-3730072)')
    parser.add_argument('--port', type=int, default=5001, help='TCP port (default: 5001)')
    parser.add_argument('--bind-addr', default=None, help='Local IP to bind to (use when two iCA202s are connected)')
    args = parser.parse_args()

    # Current state — updated incrementally as sentences arrive
    state = {col: '' for col in CSV_COLUMNS}
    state.update({'fix_quality': -1, 'num_satellites': 0,
                  'hdop': math.nan, 'height_m': math.nan,
                  'latitude_deg': math.nan, 'longitude_deg': math.nan,
                  'heading_deg': math.nan, 'speed_knots': math.nan,
                  'ant2_latitude_deg': math.nan, 'ant2_longitude_deg': math.nan,
                  'ant2_height_m': math.nan})

    print(f'Connecting to {args.host}:{args.port} ...')
    try:
        sock = socket.create_connection((args.host, args.port), timeout=10,
                                        source_address=(args.bind_addr, 0) if args.bind_addr else None)
    except OSError as e:
        print(f'Connection failed: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'Connected. Logging to {args.output}  (Ctrl+C to stop)')

    row_count = 0
    buf = ''

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        try:
            while True:
                chunk = sock.recv(4096).decode('ascii', errors='replace')
                if not chunk:
                    print('Connection closed by device.')
                    break

                buf += chunk
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()

                    if not line or not nmea_checksum_ok(line):
                        continue

                    # Strip leading $ and trailing *XX checksum for splitting
                    body = line[1:].split('*')[0]
                    fields = body.split(',')
                    sentence_id = fields[0]

                    parser_fn = PARSERS.get(sentence_id)
                    if parser_fn is None:
                        continue

                    updates = parser_fn(fields)
                    if updates:
                        state.update(updates)

                    # Write a row each time we get a fresh GGA (position epoch)
                    if sentence_id in ('GNGGA', 'GPGGA'):
                        writer.writerow(state)
                        csvfile.flush()
                        row_count += 1
                        fix_label = {4: 'RTK-fixed', 5: 'RTK-float'}.get(
                            state['fix_quality'], f"fix={state['fix_quality']}"
                        )
                        print(
                            f"\r[{row_count:5d}]  {state['timestamp_utc']}  "
                            f"lat={state['latitude_deg']:.7f}  "
                            f"lon={state['longitude_deg']:.7f}  "
                            f"h={state['height_m']:.3f}m  "
                            f"hdg={state['heading_deg']:.2f}°  "
                            f"{fix_label}",
                            end='', flush=True
                        )

        except KeyboardInterrupt:
            print(f'\nStopped. {row_count} rows written to {args.output}')
        finally:
            sock.close()


if __name__ == '__main__':
    main()
