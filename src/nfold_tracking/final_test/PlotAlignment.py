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
OUT_PNG = BASE / 'results' / f'alignment_plot_{args.mode}_{args.name}.png'

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

fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
fig.suptitle(f'{args.mode.capitalize()} Tracker vs GNSS Ground Truth — {args.name}  (Full 3D Comparison)',
             fontsize=13, fontweight='bold')

GNSS_KW    = dict(color='tab:blue',   lw=1.5, label='GNSS (ground truth)')
TRACKER_KW = dict(color='tab:orange', lw=1.5, label=f'Tracker ({args.mode.capitalize()})', alpha=0.85)
ERROR_KW   = dict(color='tab:red',    lw=1.2)


def plot_pair(ax, gnss_vals, tracker_vals, ylabel, title):
    ax.plot(t, gnss_vals,    **GNSS_KW)
    ax.plot(t, tracker_vals, **TRACKER_KW)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


for row_i, (ant2_spec, ant1_spec, row_label) in enumerate(zip(ANT2_ROWS, ANT1_ROWS, ROW_LABELS)):
    for col_i, (gnss_col, tracker_col, ylabel) in enumerate(
            [(ant2_spec[0], ant2_spec[1], ant2_spec[2]),
             (ant1_spec[0], ant1_spec[1], ant1_spec[2])]):
        ax = axes[row_i, col_i]
        gnss_vals    = valid[gnss_col]
        tracker_vals = valid[tracker_col]
        err = tracker_vals - gnss_vals
        m, s = err.mean(), err.std()

        ant_label = 'ANT2 (board marker)' if col_i == 0 else 'ANT1 (blade tip)'
        title = f'{row_label} — {ant_label}   err mean={m:+.1f} mm  std={s:.1f} mm'
        plot_pair(ax, gnss_vals, tracker_vals, ylabel, title)

# X labels on bottom row only
for ax in axes[3, :]:
    ax.set_xlabel('Video time (s)')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved → {OUT_PNG}")
plt.show()
