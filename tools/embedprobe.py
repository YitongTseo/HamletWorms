#!/usr/bin/env python3
"""Diff the C embedding forward pass against numpy's, bit for bit.

Isolates one suspect. The sim-level crosscheck diverges at tick 44461, which is
too abrupt for accumulated rounding and looks like a flipped fire decision — so
the question is whether wm_chemo_embed() and EmbeddingModel.embed_batch() agree
exactly on the same inputs. If they do not, the culprit is numpy rather than
libm: np.exp uses numpy's own SIMD kernel, not the platform exp, and `@` goes
through BLAS with blocking and FMA, whereas wm_affine sums plainly left to right.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V7 = REPO.parent / "HamletRNAWorld" / "v7"

os.environ.setdefault("NLTK_DATA", str(HERE / "nltk_data"))
os.environ["WORMLET_PASSAGE"] = "full"
os.environ["WORMLET_GENERATIONS_ENABLED"] = "1"

sys.path.insert(0, str(V7))

import numpy as np  # noqa: E402

from corpus.hamlet import get_sentences_with_flags  # noqa: E402
from server import embedding  # noqa: E402

asset = REPO / "build" / "liam.hwrm"
c_out = subprocess.run(
    [str(HERE / "hostsim" / "hostsim"), str(asset), "1", "0", "embed"],
    capture_output=True, text=True, check=True,
).stdout.strip().splitlines()

# Flatten the corpus the same way bake.py did, so token ids line up.
sentences, _ = get_sentences_with_flags("full")
tokens = [t for s in sentences for t in s]

model = embedding.EmbeddingModel(embedding.EmbeddingParams.random_init(0))

bits = lambda x: struct.pack("<d", x)
worst = 0.0
n_exact = n_diff = 0

for line in c_out:
    f = line.split()
    if f[0] != "PROBE" or f[2] == "OOV":
        continue
    tok, n_hist = int(f[1]), int(f[2])
    c_vec = [float.fromhex(x) for x in f[3:]]

    # The C probe uses probes[p][1..] as history, most-recent-first.
    idx = c_out.index(line)
    hist_toks = [[0], [100], [100, 101], [205, 100, 101, 102],
                 [311, 205, 100, 101, 102], [412, 311, 205, 100, 101]][idx][:n_hist]
    py_vec = model.embed(tokens[tok], [tokens[h] for h in hist_toks])
    if py_vec is None:
        print(f"tok {tok}: python says OOV but C produced a vector")
        continue

    same = all(bits(a) == bits(b) for a, b in zip(py_vec, c_vec))
    delta = max(abs(a - b) for a, b in zip(py_vec, c_vec))
    worst = max(worst, delta)
    n_exact += same
    n_diff += not same
    print(f"tok {tok:5d} ({tokens[tok]!r:>14}, {n_hist} hist)  "
          f"{'exact' if same else 'DIFFERS'}  max|delta|={delta:.3e}")

print(f"\n{n_exact} exact, {n_diff} differing.  worst |delta| = {worst:.3e}")
if n_diff:
    print("\nThe embedding pass is the divergence source, not libm.")
    print("np.exp is numpy's SIMD kernel and `@` is BLAS; wm_affine sums plainly.")
    print("Phase 2 makes both sides call the same kernel.")
