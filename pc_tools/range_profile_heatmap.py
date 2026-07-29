"""
range_profile_heatmap.py

Captures a fixed-duration sample from the mmWave demo's DATA UART port and
builds a range-time heatmap. Optionally prints/beeps timed breathing cues
during the capture, using the SAME clock as the recorded data -- this
removes any sync ambiguity between "when I think I breathed" and "when the
data says something happened."

Requires: pip install pyserial numpy matplotlib

Usage (with synchronized cues, recommended):
    python range_profile_heatmap.py --tlv-type 302 --max-range 11.0 \
        --window-max 1.0 --cue-times 10,20,40,50 --cue-duration 5 \
        --out-prefix breathing02

Cues print to the console (and beep) at exactly cue-time seconds after the
capture starts, using the identical clock the saved timestamps use. Breathe
when you see/hear the cue, not on your own count.
"""

import argparse
import struct
import time
import serial
import numpy as np
import matplotlib.pyplot as plt

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
HEADER_FMT = '<8sIIIIIIII'
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 40 bytes


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
    ap.add_argument('--duration', type=float, default=120.0, help='Capture length in seconds')
    ap.add_argument('--tlv-type', type=int, required=True)
    ap.add_argument('--max-range', type=float, default=11.0)
    ap.add_argument('--window-max', type=float, default=1.0)
    ap.add_argument('--out-prefix', default='capture')
    ap.add_argument('--cue-times', type=str, default='',
                     help='Comma-separated seconds (relative to capture start) to cue a breath, e.g. "10,20,40,50"')
    ap.add_argument('--cue-duration', type=float, default=5.0,
                     help='How long each cued breath is assumed to last, for shading the final plot (s)')
    args = ap.parse_args()

    cue_times = [float(x) for x in args.cue_times.split(',') if x.strip() != '']

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    print(f"Listening on {args.port} @ {args.baud} baud for {args.duration:.0f}s...")
    if cue_times:
        print(f"Cues scheduled at t={cue_times} (capture-clock seconds). Breathe when cued, not on your own count.")

    numRangeBins = None
    rangeStep = None
    bin_hi = None

    frames_db = []
    frame_times = []
    cues_fired = []

    plt.ion()
    fig, ax = plt.subplots()
    im = None
    cue_text = fig.text(0.5, 0.97, '', ha='center', va='top',
                         fontsize=22, color='red', fontweight='bold')

    t0 = time.time()
    next_cue_idx = 0
    try:
        for frameNum, tlvs in read_frames(ser):
            now = time.time() - t0
            if now >= args.duration:
                break

            while next_cue_idx < len(cue_times) and now >= cue_times[next_cue_idx]:
                print(f"\n*** CUE t={cue_times[next_cue_idx]:.1f}s: BREATHE NOW ***\a")
                cues_fired.append(cue_times[next_cue_idx])
                next_cue_idx += 1

            cue_active = any(c <= now < c + args.cue_duration for c in cue_times)
            cue_text.set_text('BREATHE NOW' if cue_active else '')

            for tlv_type, payload in tlvs:
                if tlv_type != args.tlv_type:
                    continue

                n_vals = len(payload) // 4
                if n_vals == 0:
                    continue

                if numRangeBins is None:
                    numRangeBins = n_vals
                    rangeStep = args.max_range / numRangeBins
                    bin_hi = max(1, int(round(args.window_max / rangeStep)))
                    print(f"numRangeBins={numRangeBins}  rangeStep={rangeStep:.4f} m  "
                          f"window=[0:{bin_hi}] bins (0 to {bin_hi * rangeStep:.3f} m)")

                values = np.array(struct.unpack(f'<{n_vals}I', payload), dtype=np.float64)
                window = values[:bin_hi]
                db = 20 * np.log10(np.clip(window, 1, None))

                frames_db.append(db)
                frame_times.append(now)

                if len(frames_db) < 2:
                    continue

                data = np.array(frames_db).T

                if im is None:
                    im = ax.imshow(data, aspect='auto', origin='lower',
                                    extent=[0, frame_times[-1], 0, args.window_max],
                                    cmap='viridis')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Range (m)')
                    ax.set_title(f'Range-Time Signal Strength (0-{args.window_max:.1f} m window)')
                    fig.colorbar(im, ax=ax, label='Signal Strength (dB)')
                    plt.show()
                else:
                    im.set_data(data)
                    im.set_extent([0, frame_times[-1], 0, args.window_max])
                    im.set_clim(vmin=data.min(), vmax=data.max())

                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nCapture stopped early by user.")
    finally:
        ser.close()

    print(f"\nCapture complete: {len(frames_db)} frames over {frame_times[-1] if frame_times else 0:.1f}s")

    if cue_times:
        for c in cue_times:
            ax.axvspan(c, c + args.cue_duration, color='red', alpha=0.15)
            ax.axvline(c, color='red', linestyle='--', linewidth=1)

    if frames_db:
        data = np.array(frames_db).T
        np.save(f"{args.out_prefix}_data.npy", data)
        np.save(f"{args.out_prefix}_times.npy", np.array(frame_times))
        np.save(f"{args.out_prefix}_cues.npy", np.array(cue_times))
        fig.savefig(f"{args.out_prefix}_plot.png", dpi=150)
        print(f"Saved: {args.out_prefix}_data.npy, {args.out_prefix}_times.npy, "
              f"{args.out_prefix}_cues.npy, {args.out_prefix}_plot.png")

    print("Close the plot window to exit.")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()