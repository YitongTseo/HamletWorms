#!/usr/bin/env python3
"""Validate the voice bank, and validate the decoder that will go into firmware.

Two things are checked:

  1. The packed .hvox index is coherent — every word resolves to a non-empty,
     block-aligned payload.
  2. A from-scratch IMA ADPCM decoder, written the way the C one will be, agrees
     with ffmpeg's decode sample for sample. Getting this wrong on-device sounds
     like static, and static is hard to debug through a 1.75" speaker.

Also exports a few words as .wav so you can actually listen.

    python3 tools/voxcheck.py --words sorrow,ghost,father,Denmark
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V7 = REPO.parent / "HamletRNAWorld" / "v7"

# The two tables that define IMA ADPCM. Straight out of the IMA spec; the C
# decoder will carry the same two arrays.
STEP = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767,
]
INDEX = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]


def decode_block(blk: bytes) -> list[int]:
    """One IMA-ADPCM-in-WAV block: 4-byte preamble, then nibbles, low first."""
    pred = struct.unpack("<h", blk[0:2])[0]
    index = blk[2]
    out = [pred]
    for byte in blk[4:]:
        for nib in (byte & 0x0F, byte >> 4):
            step = STEP[index]
            # ffmpeg's adpcm_ima_expand_nibble: ONE truncation, not four.
            # The classic IMA reference accumulates step>>3 + step>>2 + step>>1
            # + step and truncates each term, which for nibble 7 at step 7 gives
            # 0+1+3+7 = 11 where this gives (15*7)>>3 = 13. Mathematically the
            # same 1.875*step; only the rounding differs. ffmpeg encoded the
            # bank, so the decoder has to round ffmpeg's way or it drifts from
            # the second sample onward.
            diff = ((2 * (nib & 7) + 1) * step) >> 3
            pred = pred - diff if nib & 8 else pred + diff
            pred = max(-32768, min(32767, pred))
            index = max(0, min(88, index + INDEX[nib]))
            out.append(pred)
    return out


def decode(payload: bytes, block_align: int) -> list[int]:
    out: list[int] = []
    for i in range(0, len(payload), block_align):
        out.extend(decode_block(payload[i:i + block_align]))
    return out


def wrap_wav(payload: bytes, sr: int, block_align: int, spb: int) -> bytes:
    """Rebuild a playable IMA-ADPCM WAV around a raw payload."""
    fmt = struct.pack("<HHIIHHH", 0x11, 1, sr, sr * block_align // spb,
                      block_align, 4, 2) + struct.pack("<H", spb)
    fact = struct.pack("<I", len(payload) // block_align * spb)
    body = (b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"fact" + struct.pack("<I", 4) + fact
            + b"data" + struct.pack("<I", len(payload)) + payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=REPO / "build" / "voices.hvox")
    ap.add_argument("--words", default="sorrow,ghost,father,denmark,worm,nothing")
    ap.add_argument("--outdir", type=Path, default=REPO / "build" / "listen")
    args = ap.parse_args()

    buf = args.bank.read_bytes()
    assert buf[:4] == b"HVOX", "not a voice bank"
    version, n, sr, codec, block_align, spb = struct.unpack("<IIIIHH", buf[4:24])
    print(f"bank    {n} words, {sr} Hz, codec={'adpcm' if codec == 0 else 'opus'}, "
          f"block_align={block_align}, samples_per_block={spb}")

    index_off = 24
    idx = [struct.unpack("<II", buf[index_off + 8 * i: index_off + 8 * i + 8])
           for i in range(n)]

    empty = sum(1 for off, ln in idx if ln == 0)
    misaligned = sum(1 for off, ln in idx if ln and ln % block_align)
    total = sum(ln for _o, ln in idx)
    print(f"index   {n - empty} voiced, {empty} empty, {misaligned} not block-aligned")
    print(f"        {total:,} B of payload, mean {total / max(n - empty, 1):.0f} B/word")
    if misaligned:
        print("        (a partial final block is normal — the decoder handles it)")

    vocab = json.loads((V7 / "cache" / "corpus_nomic512.json").read_text())["words"]
    pos = {w: i for i, w in enumerate(vocab)}

    args.outdir.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for word in args.words.split(","):
        w = word.strip().lower()
        if w not in pos:
            print(f"  {w!r}: not in vocabulary")
            continue
        off, ln = idx[pos[w]]
        payload = buf[off:off + ln]

        mine = decode(payload, block_align)
        wav = wrap_wav(payload, sr, block_align, spb)
        (args.outdir / f"{w}.wav").write_bytes(wav)

        # ffmpeg is the reference: if the firmware decoder matches it, the board
        # will sound like the laptop does.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            tmp = f.name
        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", tmp, "-f", "s16le", "-acodec",
             "pcm_s16le", "-"], capture_output=True, check=True).stdout
        ref = list(struct.unpack(f"<{len(raw) // 2}h", raw))

        m = min(len(ref), len(mine))
        mism = sum(1 for a, b in zip(ref[:m], mine[:m]) if a != b)
        dur = m / sr
        if mism == 0:
            ok += 1
            print(f"  {w:<10} {ln:>6} B  {dur:.2f}s  {m} samples  decoder matches ffmpeg")
        else:
            bad += 1
            print(f"  {w:<10} {ln:>6} B  {dur:.2f}s  {mism}/{m} SAMPLES DIFFER")

    print(f"\n{ok} words decode exactly, {bad} mismatched")
    print(f"listen: open {args.outdir}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
