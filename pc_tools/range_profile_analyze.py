"""
range_profile_analyze.py

Offline analysis of a saved range-time capture (capture_data.npy +
capture_times.npy from range_profile_heatmap.py). Isolates a specific
target range (e.g. 0.5m) and plots it on its own scale, independent of
whatever near-field clutter dominated the original heatmap's color range.

Usage:
    python range_profile_analyze.py --data capture_data.npy --times capture_times.npy --target-range 0.5

    --window-max: the y-axis window used during capture (default 1.0m,
                   must match what you passed to range_profile_heatmap.py)
    --target-range: distance in meters you want to inspect (e.g. 0.5)
    --span: how many bins around the target to average/show (default 1)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to capture_data.npy (bins x frames, dB)')
    ap.add_argument('--times', required=True, help='Path to capture_times.npy')
    ap.add_argument('--window-max', type=float, default=1.0, help='Y-axis window used during capture (m)')
    ap.add_argument('--target-range', type=float, required=True, help='Distance in meters to inspect')
    ap.add_argument('--span', type=int, default=1, help='Number of bins on each side to average around target')
    args = ap.parse_args()

    data = np.load(args.data)          # shape (bins, frames), dB
    times = np.load(args.times)        # shape (frames,), seconds

    n_bins = data.shape[0]
    range_step = args.window_max / n_bins
    target_bin = int(round(args.target_range / range_step))
    target_bin = max(0, min(n_bins - 1, target_bin))

    lo = max(0, target_bin - args.span)
    hi = min(n_bins, target_bin + args.span + 1)

    print(f"n_bins={n_bins}  range_step={range_step:.4f} m  "
          f"target_bin={target_bin} ({target_bin*range_step:.3f} m)  "
          f"averaging bins [{lo}:{hi}) -> {lo*range_step:.3f}-{hi*range_step:.3f} m")

    trace = data[lo:hi, :].mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].imshow(data, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], 0, args.window_max],
                    cmap='viridis')
    axes[0].axhline(target_bin * range_step, color='red', linestyle='--', linewidth=1,
                     label=f'target ~{args.target_range:.2f} m')
    axes[0].set_ylabel('Range (m)')
    axes[0].set_title('Full window (original color scale, for reference)')
    axes[0].legend(loc='upper right')

    axes[1].plot(times, trace, linewidth=1)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Signal Strength (dB)')
    axes[1].set_title(f'Isolated trace at ~{args.target_range:.2f} m (own scale)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'target_range_trace.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == '__main__':
    main()