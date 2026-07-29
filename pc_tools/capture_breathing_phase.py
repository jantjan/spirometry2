"""
capture_breathing_phase.py

Reads the "PH,<frame>,<bin>:<phase>,..." debug stream emitted by the
BREATHING_DEBUG_STREAM block in dpc.c, and plots raw (wrapped) and
unwrapped phase for each candidate range bin.

Usage:
    pip install pyserial matplotlib numpy
    python capture_breathing_phase.py

Before running: close any other program (visualizer, terminal) that has
the COM port open -- only one process can hold a serial port at a time.
"""

import time
import serial
import numpy as np
import matplotlib.pyplot as plt

# ---- EDIT THESE ----
PORT = "COM6"          # your Application/User UART COM port
BAUD = 115200
DURATION_S = 30         # how many seconds to capture (aim for several full breaths)
# ---------------------

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Listening on {PORT} @ {BAUD} for {DURATION_S}s... breathe normally at ~1m from the sensor.")

    frames = []
    bins_seen = None
    phase_by_bin = None  # dict: bin_index -> list of raw phase values

    t_end = time.time() + DURATION_S
    while time.time() < t_end:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw.startswith("PH,"):
            continue

        parts = raw.split(",")
        try:
            frame_num = int(parts[1])
        except ValueError:
            continue

        entries = []
        for tok in parts[2:]:
            if ":" not in tok:
                continue
            bin_str, phase_str = tok.split(":")
            try:
                entries.append((int(bin_str), float(phase_str)))
            except ValueError:
                continue

        if not entries:
            continue

        if bins_seen is None:
            bins_seen = [b for b, _ in entries]
            phase_by_bin = {b: [] for b in bins_seen}

        frames.append(frame_num)
        for b, p in entries:
            if b in phase_by_bin:
                phase_by_bin[b].append(p)

    ser.close()

    if not frames:
        print("No data captured. Check COM port, baud rate, and that the sensor is streaming.")
        return

    print(f"Captured {len(frames)} frames across bins: {bins_seen}")

    n_bins = len(bins_seen)
    fig, axes = plt.subplots(n_bins, 2, figsize=(12, 2.2 * n_bins), sharex=True)
    if n_bins == 1:
        axes = [axes]

    for row, b in enumerate(bins_seen):
        raw_phase = np.array(phase_by_bin[b])
        n = min(len(frames), len(raw_phase))
        x = frames[:n]
        raw_phase = raw_phase[:n]
        unwrapped = np.unwrap(raw_phase)
        # remove linear/DC trend just for easier visual inspection
        unwrapped_detrended = unwrapped - np.linspace(unwrapped[0], unwrapped[-1], len(unwrapped))

        ax_raw, ax_unw = axes[row]
        ax_raw.plot(x, raw_phase, linewidth=0.8)
        ax_raw.set_ylabel(f"bin {b}\nraw (rad)")
        ax_raw.set_ylim(-np.pi - 0.5, np.pi + 0.5)

        ax_unw.plot(x, unwrapped_detrended, linewidth=0.8, color="darkorange")
        ax_unw.set_ylabel("unwrapped\n(detrended, rad)")

        if row == 0:
            ax_raw.set_title("Raw wrapped phase")
            ax_unw.set_title("Unwrapped phase (drift removed)")

    axes[-1][0].set_xlabel("frame #")
    axes[-1][1].set_xlabel("frame #")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()