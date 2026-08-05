#!/usr/bin/env python3
"""Render every word in the worm's vocabulary and pack it into one flash blob.

macOS `say` is the synthesiser — it is offline, high quality, and already on the
machine. Daniel is en_GB, which is the right accent for Hamlet.

Codec choice is measured, not guessed. Over a 59-word sample of the real vocab
(rendered, silence-trimmed, and encoded five ways), extrapolated to 4919 words:

    Opus 16 kbps @16 kHz    990 B/word     4.65 MB
    Opus 24 kbps @16 kHz   1372 B/word     6.44 MB
    MP3  32 kbps           2228 B/word    10.45 MB
    IMA-ADPCM  @8 kHz      2263 B/word    10.62 MB   <- default
    IMA-ADPCM  @11.025 kHz 2888 B/word    13.55 MB

ADPCM wins on integration risk, not on size: the decoder is about 30 lines and
costs essentially no CPU, and 10.6 MB fits the 12 MB data partition alongside
the 1 MB worm asset. Opus halves the size and sounds better through the little
speaker, but drags in libopus. `--codec opus` is wired up for when that trade
looks worth making.

Mean trimmed word is 0.46 s, so the worm can speak about two words a second,
which is roughly the rate it eats at.

    python3 tools/voices.py --out build/voices.hvox -j 6
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V7 = REPO.parent / "HamletRNAWorld" / "v7"

MAGIC = b"HVOX"
VERSION = 1

# Peak-detect rather than RMS: `say` leaves a chunk of digital silence at both
# ends, and trimming it is most of the size win.
TRIM = ("silenceremove=start_periods=1:start_threshold=-40dB:"
        "start_silence=0.01:detection=peak,areverse,"
        "silenceremove=start_periods=1:start_threshold=-40dB:"
        "start_silence=0.01:detection=peak,areverse")

# Loudness is fixed here, in the bank, not with a gain knob at playback.
# `say` output peaks around half full scale but averages far below it — a crest
# factor of roughly 8 — so simply multiplying on the way to the codec clips the
# peaks off 85% of words without making them meaningfully louder. speechnorm is
# built for exactly this: it lifts speech toward full scale envelope by
# envelope, and the limiter catches what is left. Measured over a sample:
# RMS 1.7x to 3.9x higher, peaks still under full scale. The quiet words gain
# the most, which is the point — "ghost" was half the level of "sorrow".
# A presence lift before normalising. At 8 kHz the whole consonant range is
# crowded into the top octave, and /s/ /t/ /f/ are what carry a word apart from
# its neighbours — boosting there does more for making the worm intelligible
# than slowing it down does.
PRESENCE = "highshelf=f=1700:g=4"

NORMALIZE = PRESENCE + ",speechnorm=e=12.5:r=0.0001:l=1,alimiter=limit=0.97"


def render_one(args) -> tuple[int, bytes, dict]:
    idx, word, voice, rate, codec, sr, tmpdir = args
    stem = Path(tmpdir) / f"w{idx:05d}"
    aiff = stem.with_suffix(".aiff")

    # `say` fails intermittently under parallel load; one retry clears it.
    for attempt in range(3):
        r = subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), word],
                           capture_output=True)
        if r.returncode == 0 and aiff.exists() and aiff.stat().st_size > 0:
            break
    else:
        return idx, b"", {"error": "say failed"}

    if codec == "adpcm":
        out = stem.with_suffix(".wav")
        enc = ["-c:a", "adpcm_ima_wav"]
    else:
        out = stem.with_suffix(".opus")
        enc = ["-c:a", "libopus", "-b:a", "24k", "-vbr", "on", "-application", "audio"]

    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-af", TRIM + "," + NORMALIZE,
         "-ar", str(sr), "-ac", "1", *enc, str(out)],
        capture_output=True, check=False,
    )
    if not out.exists():
        return idx, b"", {"error": "encode failed"}

    data = out.read_bytes()
    meta: dict = {}
    if codec == "adpcm":
        data, meta = strip_wav(data)
    aiff.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return idx, data, meta


def strip_wav(buf: bytes) -> tuple[bytes, dict]:
    """Pull the raw ADPCM payload plus the two numbers a decoder actually needs.

    IMA ADPCM in a WAV is block-structured: each block opens with a 4-byte
    preamble (initial predictor and step index) and the rest is 4-bit nibbles.
    The firmware decoder needs block_align and samples_per_block, and they are
    the same for every file at a given sample rate, so they are stored once in
    the bank header rather than per word.
    """
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return b"", {"error": "not a wav"}
    pos = 12
    meta: dict = {}
    payload = b""
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4]
        size = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
        body = buf[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            meta["block_align"] = struct.unpack("<H", body[12:14])[0]
            if len(body) >= 20:
                meta["samples_per_block"] = struct.unpack("<H", body[18:20])[0]
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    return payload, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="Daniel", help="macOS voice; Daniel is en_GB")
    ap.add_argument("--rate", type=int, default=180, help="words per minute")
    ap.add_argument("--codec", choices=("adpcm", "opus"), default="adpcm")
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("-j", "--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N words (smoke test)")
    ap.add_argument("--out", type=Path, default=REPO / "build" / "voices.hvox")
    args = ap.parse_args()

    nomic = json.loads((V7 / "cache" / "corpus_nomic512.json").read_text())
    vocab = nomic["words"]
    if args.limit:
        vocab = vocab[: args.limit]
    print(f"voice   {args.voice} @ {args.rate} wpm, {args.codec} {args.sr} Hz")
    print(f"vocab   {len(vocab)} words, {args.jobs} workers")

    blobs: list[bytes] = [b""] * len(vocab)
    fmt: dict = {}
    errors = 0

    with tempfile.TemporaryDirectory() as tmp:
        work = [(i, w, args.voice, args.rate, args.codec, args.sr, tmp)
                for i, w in enumerate(vocab)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for done, (idx, data, meta) in enumerate(ex.map(render_one, work), 1):
                blobs[idx] = data
                if meta.get("error") or not data:
                    errors += 1
                elif not fmt and "block_align" in meta:
                    fmt = meta
                if done % 250 == 0 or done == len(vocab):
                    got = sum(len(b) for b in blobs)
                    print(f"  {done:5d}/{len(vocab)}  {got/1048576:6.2f} MB  {errors} errors",
                          flush=True)

    # --- pack ---------------------------------------------------------------
    # HVOX | version | n | sample_rate | codec | block_align | samples_per_block
    # then n x (u32 offset, u32 length), then the payloads. Index order is the
    # vocab order, which is the same integer the worm asset uses, so the
    # firmware speaks a word straight from its vocab id with no string lookup.
    head = bytearray(MAGIC)
    head += struct.pack("<IIIIHH", VERSION, len(vocab), args.sr,
                        0 if args.codec == "adpcm" else 1,
                        fmt.get("block_align", 0), fmt.get("samples_per_block", 0))
    index_off = len(head)
    body_off = index_off + 8 * len(vocab)

    index = bytearray()
    body = bytearray()
    cur = body_off
    for b in blobs:
        index += struct.pack("<II", cur if b else 0, len(b))
        body += b
        cur += len(b)

    out = bytes(head) + bytes(index) + bytes(body)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(out)

    spoken = sum(1 for b in blobs if b)
    print(f"\n  {spoken}/{len(vocab)} words voiced ({errors} failed)")
    print(f"  block_align={fmt.get('block_align')} samples_per_block={fmt.get('samples_per_block')}")
    print(f"  {len(out):,} B ({len(out)/1048576:.2f} MB) -> {args.out}")
    if spoken:
        print(f"  mean {sum(len(b) for b in blobs)/spoken:.0f} B/word")


if __name__ == "__main__":
    main()
