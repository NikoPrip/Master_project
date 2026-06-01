"""
PlotAlignment.py — Full 3D comparison: tracker vs GNSS for ANT2 and ANT1.

Layout: 4 rows × 2 cols
  Rows: dE, dN, dU, 3D distance
  Cols: ANT2 (board marker, near)  |  ANT1 (blade tip, far)

ANT2 tracker is in ENU via Wahba-fitted R_cam_ENU (fitted on ANT2 data itself).
ANT1 tracker is in ENU via the same rotation (forward-applied, not re-fitted).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

parser = argparse.ArgumentParser(description='Plot tracker vs GNSS alignment.')
parser.add_argument('--name', default='output',
                    help="Session name matching AlignGNSS.py --name (default: output)")
parser.add_argument('--mode', default='aruco',
                    choices=['aruco', 'hybrid', 'nfold'],
                    help='Tracker mode (default: aruco)')
args    = parser.parse_args()
IN_CSV  = BASE / 'results' / f'aligned_{args.mode}_{args.name}.csv'
OUT_NEAR = BASE / 'results' / f'alignment_plot_{args.mode}_{args.name}_near.png'
OUT_FAR  = BASE / 'results' / f'alignment_plot_{args.mode}_{args.name}_far.png'

df    = pd.read_csv(IN_CSV)
valid = df.dropna(subset=[
    'gnss_dist_mm', 'tracker_dist_mm',
    'gnss_ant1_dist_mm', 'tracker_ant1_dist_mm',
    'tracker_ant2_dE_mm', 'tracker_ant2_dN_mm', 'tracker_ant2_dU_mm',
    'tracker_ant1_dE_mm', 'tracker_ant1_dN_mm', 'tracker_ant1_dU_mm',
]).copy()
t = valid['time_s']

# ---------------------------------------------------------------------------
# Component labels and column pairs: (gnss_col, tracker_col, ylabel)
# ---------------------------------------------------------------------------

ANT2_ROWS = [
    ('gnss_dE_mm',       'tracker_ant2_dE_mm', 'dE  (mm)'),
    ('gnss_dN_mm',       'tracker_ant2_dN_mm', 'dN  (mm)'),
    ('gnss_dU_mm',       'tracker_ant2_dU_mm', 'dU  (mm)'),
    ('gnss_dist_mm',     'tracker_dist_mm',    'Distance (mm)'),
]

ANT1_ROWS = [
    ('gnss_ant1_dE_mm',  'tracker_ant1_dE_mm', 'dE  (mm)'),
    ('gnss_ant1_dN_mm',  'tracker_ant1_dN_mm', 'dN  (mm)'),
    ('gnss_ant1_dU_mm',  'tracker_ant1_dU_mm', 'dU  (mm)'),
    ('gnss_ant1_dist_mm','tracker_ant1_dist_mm','Distance (mm)'),
]

ROW_LABELS = ['East component', 'North component', 'Up component', '3D distance']

# ---------------------------------------------------------------------------
# Plot: 4 rows × 2 cols
# ---------------------------------------------------------------------------

_tracker_labels = {'aruco': 'ArUco', 'hybrid': 'Hybrid', 'nfold': 'Nfold'}
_tracker_label  = _tracker_labels.get(args.mode, args.mode.capitalize())
_surface        = ('Terrain: Incline' if 'dirt_full' in args.name or 'incline' in args.name else
                   'Terrain: Dirt'   if 'dirt' in args.name else
                   'Terrain: Brick')

GNSS_KW    = dict(color='tab:blue',   lw=1.5, label='GNSS (ground truth)')
TRACKER_KW = dict(color='tab:orange', lw=1.5, label=f'Tracker ({_tracker_label})', alpha=0.85)


def plot_column(rows, ant_label, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for row_i, (gnss_col, tracker_col, ylabel) in enumerate(rows):
        ax = axes[row_i // 2, row_i % 2]
        gnss_vals    = valid[gnss_col]
        tracker_vals = valid[tracker_col]
        err = tracker_vals - gnss_vals
        m, s = err.mean(), err.std()
        ax.plot(t, gnss_vals,    **GNSS_KW)
        ax.plot(t, tracker_vals, **TRACKER_KW)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'{ROW_LABELS[row_i]} — {ant_label}   err mean={m:+.1f} mm  std={s:.1f} mm', fontsize=9)
        ax.set_xlabel('Video time (s)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


plot_column(ANT2_ROWS, 'ANT2 (board marker)',
            f'{_tracker_label} Tracker vs GNSS — Near Antenna (Board Marker) - {_surface}',
            OUT_NEAR)

plot_column(ANT1_ROWS, 'ANT1 (blade tip)',
            f'{_tracker_label} Tracker vs GNSS — Far Antenna (Blade Tip) - {_surface}',
            OUT_FAR)
