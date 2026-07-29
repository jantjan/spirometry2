"""
range_profile_zoom_events.py

Offline analysis step 2: average signal strength over a chosen range BAND
(not just one bin) across a chosen time window, then run basic rising/
falling edge detection to flag candidate breath-onset events.

Requires: pip install numpy matplotlib scipy

Usage:
    python range_profile_zoom_events.py --data breathing_data.npy --times breathing_times.npy \
        --range-lo 0.4 --range-hi 0.75 --time-lo 25 --time-hi 40

NOTE: rising vs falling edges here are just "signal getting stronger" vs
"getting weaker" -- we don't yet know which corresponds to inhale vs
exhale. You'll need to correlate the printed event timestamps against a
known breath timing (e.g. count out loud on video, or press a key at the
start of each inhale/exhale while recording) to figure out the mapping.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--times', required=True)
    ap.add_argument('--window-max', type=float, default=1.0, help='Y-axis window used during capture (m)')
    ap.add_argument('--range-lo', type=float, required=True, help='Lower bound of range band (m)')
    ap.add_argument('--range-hi', type=float, required=True, help='Upper bound of range band (m)')
    ap.add_argument('--time-lo', type=float, default=None, help='Zoom start time (s), default = full capture')
    ap.add_argument('--time-hi', type=float, default=None, help='Zoom end time (s), default = full capture')
    ap.add_argument('--smooth-win', type=int, default=7, help='Savitzky-Golay smoothing window (odd, frames)')
    args = ap.parse_args()

    data = np.load(args.data)      # (bins, frames)
    times = np.load(args.times)    # (frames,)

    n_bins = data.shape[0]
    range_step = args.window_max / n_bins
    bin_lo = max(0, int(round(args.range_lo / range_step)))
    bin_hi = min(n_bins, int(round(args.range_hi / range_step)))
    print(f"range_step={range_step:.4f} m  band bins [{bin_lo}:{bin_hi}) "
          f"-> {bin_lo*range_step:.3f}-{bin_hi*range_step:.3f} m")

    trace = data[bin_lo:bin_hi, :].mean(axis=0)

    t_lo = args.time_lo if args.time_lo is not None else times[0]
    t_hi = args.time_hi if args.time_hi is not None else times[-1]
    mask = (times >= t_lo) & (times <= t_hi)
    t_zoom = times[mask]
    trace_zoom = trace[mask]

    win = args.smooth_win if args.smooth_win % 2 == 1 else args.smooth_win + 1
    win = min(win, len(trace_zoom) - (1 - len(trace_zoom) % 2))
    smoothed = savgol_filter(trace_zoom, window_length=max(5, win), polyorder=2) if len(trace_zoom) > 5 else trace_zoom

    deriv = np.gradient(smoothed, t_zoom)

    rising_idx, _ = find_peaks(deriv, height=np.std(deriv) * 0.5)
    falling_idx, _ = find_peaks(-deriv, height=np.std(deriv) * 0.5)

    print("\nCandidate 'signal increasing fastest' events (rising edges):")
    for i in rising_idx:
        print(f"  t={t_zoom[i]:6.2f}s  dB={smoothed[i]:6.2f}")

    print("\nCandidate 'signal decreasing fastest' events (falling edges):")
    for i in falling_idx:
        print(f"  t={t_zoom[i]:6.2f}s  dB={smoothed[i]:6.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t_zoom, trace_zoom, alpha=0.4, label='raw')
    axes[0].plot(t_zoom, smoothed, linewidth=2, label='smoothed')
    axes[0].scatter(t_zoom[rising_idx], smoothed[rising_idx], color='green', zorder=5, label='rising edge')
    axes[0].scatter(t_zoom[falling_idx], smoothed[falling_idx], color='red', zorder=5, label='falling edge')
    axes[0].set_ylabel('Signal Strength (dB)')
    axes[0].set_title(f'Range band {args.range_lo}-{args.range_hi} m, t={t_lo:.1f}-{t_hi:.1f}s')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_zoom, deriv)
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('d(dB)/dt')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('zoom_events.png', dpi=150)
    print("\nSaved zoom_events.png")
    plt.show()


if __name__ == '__main__':
    main()