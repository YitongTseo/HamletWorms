#!/usr/bin/env python3
"""Verify the firmware's genome-parse algorithm against a real weights.json.

sync.c resolves every edge by NAME — binary search into the sorted neuron table
for the pre-synaptic neuron, then a scan of that CSR row for the post-synaptic
column — rather than trusting the downloaded file to enumerate in the same order
bake.py happened to see. This replicates that exactly and checks two things:

  1. every edge in the file lands in a slot, and every slot is filled once
  2. the resulting array is bit-identical to the baked one

The second is the real test. If reading gen-0007's own weights.json back through
the download path does not reproduce the table baked from that same file, then
a downloaded genome would quietly wire the connectome wrong — every weight
present, none of them where they belong.

    python3 tools/synccheck.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V7 = REPO.parent / "HamletRNAWorld" / "v7"


def read_asset(path: Path):
    buf = path.read_bytes()
    assert buf[:4] == b"HWRM", "not an asset"
    n_sections = struct.unpack("<I", buf[8:12])[0]
    sections = {}
    for i in range(n_sections):
        e = 16 + 16 * i
        tag = buf[e:e + 8].decode().strip()
        off, ln = struct.unpack("<II", buf[e + 8:e + 16])
        sections[tag] = buf[off:off + ln]

    def strtab(b):
        count, blob_len = struct.unpack("<II", b[:8])
        offs = struct.unpack(f"<{count+1}I", b[8:8 + 4 * (count + 1)])
        blob = b[8 + 4 * (count + 1):8 + 4 * (count + 1) + blob_len]
        return [blob[offs[i]:offs[i + 1]].decode() for i in range(count)], \
               8 + 4 * (count + 1) + blob_len

    neurons, used = strtab(sections["NEURONS"])

    syn = sections["SYNAPSE"]
    n_neurons, n_edges = struct.unpack("<II", syn[:8])
    p = 8
    row_start = struct.unpack(f"<{n_neurons+1}I", syn[p:p + 4 * (n_neurons + 1)])
    p += 4 * (n_neurons + 1)
    cols = struct.unpack(f"<{n_edges}H", syn[p:p + 2 * n_edges])
    p += 2 * n_edges
    p = (p + 7) // 8 * 8
    w = struct.unpack(f"<{n_edges}d", syn[p:p + 8 * n_edges])
    return neurons, row_start, cols, w


def bsearch(neurons, name):
    """The C does a binary search over the sorted table; mirror it exactly."""
    lo, hi = 0, len(neurons) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if neurons[mid] == name:
            return mid
        if name < neurons[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


def main() -> None:
    asset = REPO / "build" / "liam.hwrm"
    if not asset.exists():
        sys.exit("run tools/bake.py first")
    neurons, row_start, cols, baked = read_asset(asset)
    print(f"asset   {len(neurons)} neurons, {len(baked)} edges")

    assert neurons == sorted(neurons), "neuron table is not sorted; bsearch would fail"

    src = V7 / "data/poetry-2/generations/flask_1/gen-0007/Liam/weights.json"
    weights = json.loads(src.read_text())

    out = list(baked)          # sync.c starts from the baked table
    filled = [0] * len(baked)
    matched = missed = 0

    for pre, targets in weights.items():
        pi = bsearch(neurons, pre)
        if pi < 0:
            missed += 1
            continue
        s, e = row_start[pi], row_start[pi + 1]
        for post, value in targets.items():
            qi = bsearch(neurons, post)
            if qi < 0:
                missed += 1
                continue
            for k in range(s, e):
                if cols[k] == qi:
                    out[k] = value
                    filled[k] += 1
                    matched += 1
                    break
            else:
                missed += 1

    print(f"resolve {matched}/{len(baked)} edges matched, {missed} unresolved")
    dup = sum(1 for f in filled if f > 1)
    empty = sum(1 for f in filled if f == 0)
    print(f"slots   {dup} written twice, {empty} never written")

    exact = sum(1 for a, b in zip(out, baked)
                if struct.pack("<d", a) == struct.pack("<d", b))
    print(f"values  {exact}/{len(baked)} bit-identical to the baked table")

    ok = matched == len(baked) and missed == 0 and dup == 0 and empty == 0 \
        and exact == len(baked)
    print("\n" + ("PASS: a downloaded genome lands exactly where the baked one is."
                  if ok else
                  "FAIL: the download path would wire the connectome wrong."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
