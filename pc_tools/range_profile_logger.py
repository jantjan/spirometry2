"""
range_profile_logger.py

Reads the mmWave demo's DATA UART port, parses the TLV packet stream, and
prints the signal strength (dB) at one chosen range bin over time.

--- Two-phase workflow ---

1) DISCOVERY MODE (find the correct TLV type + bin index for your setup):

    python range_profile_logger.py --port COM6 --baud 1250000 --bin -1

   This prints the type/length of every TLV in every frame. Look for the one
   whose length == 4 * numRangeBins -- that's your range profile TLV type.

2) LOGGING MODE (once you know the TLV type and target bin):

    python range_profile_logger.py --port COM6 --baud 1250000 --tlv-type <N> --bin <idx>
"""

import argparse
import struct
import time
import serial
import numpy as np

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
    ap.add_argument('--port', default='COM6', help='Data COM port')
    ap.add_argument('--baud', type=int, default=1250000)
    ap.add_argument('--tlv-type', type=int, default=None, help='Range Profile TLV type (see discovery mode)')
    ap.add_argument('--bin', type=int, default=-1, help='Range bin index to track. -1 = discovery mode')
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    print(f"Listening on {args.port} @ {args.baud} baud... (Ctrl+C to stop)")

    t0 = time.time()
    try:
        for frameNum, tlvs in read_frames(ser):
            for tlv_type, payload in tlvs:
                n_vals = len(payload) // 4

                if args.bin == -1:
                    print(f"frame {frameNum:6d}  TLV type={tlv_type:3d}  "
                          f"len={len(payload):6d}  (n_uint32={n_vals})")
                    continue

                if args.tlv_type is not None and tlv_type != args.tlv_type:
                    continue
                if n_vals == 0 or args.bin >= n_vals:
                    continue

                values = struct.unpack(f'<{n_vals}I', payload)
                mag = values[args.bin]
                db = 20 * np.log10(mag) if mag > 0 else float('-inf')
                t = time.time() - t0
                print(f"{t:8.3f}s  frame={frameNum:6d}  bin={args.bin:3d}  raw={mag:10d}  ~{db:6.2f} dB")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()