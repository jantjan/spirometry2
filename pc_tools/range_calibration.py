"""
range_calibration.py

Empirical range calibration test. Prompts you (on-screen, big red text) to
hold a hand/flat object at a sequence of KNOWN distances, one at a time,
each for a fixed hold-time. Plots the RAW range profile by BIN INDEX (not
meters -- we deliberately do NOT assume any meters conversion here, since
that's the exact thing being tested). At the end, it automatically finds
the bin that lit up most during each known-distance window and fits a
straight line (distance = rangeStep * bin + offset) to recover the real,
current bin-to-meters mapping.

Requires: pip install pyserial numpy matplotlib

Usage:
    python range_calibration.py --tlv-type 303 --distances 0.3,0.6,0.9 --hold-time 10

(port/baud default to COM6 / 1250000, same as your other scripts)

How to run the physical test:
    1. Start the script.
    2. When the on-screen text says "HOLD AT 0.30 m", hold your flat hand
       (palm facing the radar) steady at exactly that distance, directly
       in front of the sensor, for the full hold-time window.
    3. Move to the next distance as soon as the text updates. Try to keep
       your hand still WITHIN each window and clearly moving BETWEEN
       windows, so the transitions are sharp in the data.
    4. Repeat for all distances in --distances, in order.
"""

import argparse
import struct
import time
import serial
import numpy as np
import matplotlib.pyplot as plt

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
HEADER_FMT = '<8sIIIIIIII'
HEADER_LEN = struct.calcsize(HEADER_FMT)


def read_frames(ser):
    buf = bytearray()
    while True:
        buf += ser.read(4096)

        idx = bytes(buf).find(MAGIC_WORD)
        if idx < 0:
            if len(buf) > 8:
                del buf[:-8]
            continue
        del buf[:idx]

        if len(buf) < HEADER_LEN:
            continue

        magic, version, totalLen, platform, frameNum, timeCpu, numDetObj, numTLV, subFrame = \
            struct.unpack(HEADER_FMT, bytes(buf[:HEADER_LEN]))

        if totalLen < HEADER_LEN or totalLen > 500000:
            del buf[:8]
            continue

        while len(buf) < totalLen:
            chunk = ser.read(totalLen - len(buf))
            if not chunk:
                break
            buf += chunk
        if len(buf) < totalLen:
            continue

        packet = bytes(buf[:totalLen])
        del buf[:totalLen]

        offset = HEADER_LEN
        tlvs = []
        for _ in range(numTLV):
            if offset + 8 > len(packet):
                break
            tlv_type, tlv_len = struct.unpack('<II', packet[offset:offset + 8])
            payload = packet[offset + 8: offset + 8 + tlv_len]
            tlvs.append((tlv_type, payload))
            offset += 8 + tlv_len

        yield frameNum, tlvs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM6')
    ap.add_argument('--baud', type=int, default=1250000)
    ap.add_argument('--tlv-type', type=int, required=True)
    ap.add_argument('--distances', type=str, required=True,
                     help='Comma-separated known distances in meters, e.g. "0.3,0.6,0.9"')
    ap.add_argument('--hold-time', type=float, default=10.0, help='Seconds to hold at each distance')
    ap.add_argument('--out-prefix', default='calibration')
    args = ap.parse_args()

    distances = [float(x) for x in args.distances.split(',') if x.strip() != '']
    schedule = []  # list of (distance, start_t, end_t)
    t_cursor = 3.0  # small settle buffer before first instruction
    for dist in distances:
        schedule.append((dist, t_cursor, t_cursor + args.hold_time))
        t_cursor += args.hold_time
    total_duration = t_cursor + 2.0

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    print(f"Listening on {args.port} @ {args.baud} baud for {total_duration:.0f}s...")
    print("Schedule:")
    for dist, s, e in schedule:
        print(f"  hold at {dist:.2f} m  from t={s:.1f}s to t={e:.1f}s")

    numRangeBins = None
    frames_all = []   # list of full-length profile arrays (raw dB, all bins)
    frame_times = []

    plt.ion()
    fig, ax = plt.subplots()
    im = None
    cue_text = fig.text(0.5, 0.97, '', ha='center', va='top',
                         fontsize=20, color='red', fontweight='bold')

    t0 = time.time()
    try:
        for frameNum, tlvs in read_frames(ser):
            now = time.time() - t0
            if now >= total_duration:
                break

            active = next(((d, s, e) for d, s, e in schedule if s <= now < e), None)
            cue_text.set_text(f'HOLD AT {active[0]:.2f} m' if active else '')

            for tlv_type, payload in tlvs:
                if tlv_type != args.tlv_type:
                    continue
                n_vals = len(payload) // 4
                if n_vals == 0:
                    continue
                if numRangeBins is None:
                    numRangeBins = n_vals
                    print(f"numRangeBins={numRangeBins} (plotting by raw bin index, no meters assumed)")

                values = np.array(struct.unpack(f'<{n_vals}I', payload), dtype=np.float64)
                db = 20 * np.log10(np.clip(values, 1, None))

                frames_all.append(db)
                frame_times.append(now)

                if len(frames_all) < 2:
                    continue

                data = np.array(frames_all).T  # (bins, frames)

                if im is None:
                    im = ax.imshow(data, aspect='auto', origin='lower',
                                    extent=[0, frame_times[-1], 0, numRangeBins],
                                    cmap='viridis')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Raw bin index')
                    ax.set_title('Range-Time (RAW BIN INDEX -- calibration in progress)')
                    fig.colorbar(im, ax=ax, label='Signal Strength (dB)')
                    plt.show()
                else:
                    im.set_data(data)
                    im.set_extent([0, frame_times[-1], 0, numRangeBins])
                    im.set_clim(vmin=data.min(), vmax=data.max())

                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        ser.close()

    print(f"\nCapture complete: {len(frames_all)} frames over {frame_times[-1] if frame_times else 0:.1f}s")

    if not frames_all:
        print("No data captured -- nothing to calibrate.")
        return

    data = np.array(frames_all).T  # (bins, frames)
    times_arr = np.array(frame_times)

    np.save(f"{args.out_prefix}_data.npy", data)
    np.save(f"{args.out_prefix}_times.npy", times_arr)
    np.save(f"{args.out_prefix}_schedule.npy", np.array(schedule))

    # --- Auto-calibration ---
    global_median = np.median(data, axis=1)  # per-bin baseline across whole session
    fitted_bins = []
    fitted_dists = []
    print("\nPer-distance detected bin (most anomalously elevated bin during that window):")
    for dist, s, e in schedule:
        m = (times_arr >= s) & (times_arr < e)
        if m.sum() == 0:
            print(f"  {dist:.2f} m: no frames in window, skipped")
            continue
        window_mean = data[:, m].mean(axis=1)
        deviation = window_mean - global_median
        best_bin = int(np.argmax(deviation))
        fitted_bins.append(best_bin)
        fitted_dists.append(dist)
        for b, s, e in schedule:
            ax.axvline(s, color='red', linestyle='--', linewidth=1)
        print(f"  {dist:.2f} m -> bin {best_bin}  (deviation {deviation[best_bin]:.1f} dB above session median)")

    if len(fitted_bins) >= 2:
        A = np.vstack([fitted_bins, np.ones(len(fitted_bins))]).T
        slope, intercept = np.linalg.lstsq(A, fitted_dists, rcond=None)[0]
        pred = slope * np.array(fitted_bins) + intercept
        resid = np.array(fitted_dists) - pred
        print("\n=== CALIBRATION RESULT ===")
        print(f"distance_m = {slope:.5f} * bin_index + {intercept:.5f}")
        print(f"i.e. rangeStep (bin spacing) = {slope:.5f} m/bin")
        print(f"residuals (m): {resid}")
        print(f"max abs residual: {np.max(np.abs(resid)):.3f} m")
        with open(f"{args.out_prefix}_fit.txt", "w") as f:
            f.write(f"distance_m = {slope:.5f} * bin_index + {intercept:.5f}\n")
            f.write(f"rangeStep = {slope:.5f} m/bin\n")
        print(f"Saved {args.out_prefix}_fit.txt")
    else:
        print("\nNot enough valid windows to fit a calibration line (need >= 2).")

    fig.savefig(f"{args.out_prefix}_plot.png", dpi=150)
    print(f"Saved: {args.out_prefix}_data.npy, {args.out_prefix}_times.npy, "
          f"{args.out_prefix}_schedule.npy, {args.out_prefix}_plot.png")

    print("Close the plot window to exit.")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()