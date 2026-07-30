#!/usr/bin/env python3
"""Run the Python v7 sim and the C port on the same worm and diff them.

This is the test that decides whether a board is a lock-step replica of its
server-side twin or merely a worm of the same lineage. It reports the first tick
at which the two disagree, and on what.

Both sides must see identical inputs: same seed, same connectome, same embedder,
same corpus pass. World reads WORMLET_PASSAGE / WORMLET_GENERATIONS_ENABLED from
the environment, so those are set here rather than left to the shell.

    python3 tools/crosscheck.py --ticks 20000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V7 = REPO.parent / "HamletRNAWorld" / "v7"

os.environ.setdefault("NLTK_DATA", str(HERE / "nltk_data"))
# Must be set before sim.world is imported: World.__post_init__ reads them to
# choose the passage and whether the scroller loops.
os.environ["WORMLET_PASSAGE"] = "full"
os.environ["WORMLET_GENERATIONS_ENABLED"] = "1"  # one pass, no looping

sys.path.insert(0, str(V7))

import numpy as np  # noqa: E402

from server import embedding  # noqa: E402
from sim.world import World  # noqa: E402


def run_python(ticks: int, state_every: int, gen_dir: Path, seed: int,
               embedder_seed: int) -> list[str]:
    weights = json.loads((gen_dir / "weights.json").read_text())
    model = embedding.EmbeddingModel(embedding.EmbeddingParams.random_init(embedder_seed))
    w = World(seed=seed, weights=weights, embedding_model=model)

    out: list[str] = []
    for t in range(ticks):
        w.tick()
        for line_id, word_idx, word in w.drain_eaten_words():
            out.append(f"EAT {w.tick_count} {line_id} {word_idx} {word}")
        if state_every > 0 and (t % state_every) == 0:
            b = w.worm
            out.append(
                f"STATE {w.tick_count} {b.target_x.hex()} {b.target_y.hex()} "
                f"{b.facing_dir.hex()} {float(b.speed).hex()}"
            )
    out.append(f"END {w.tick_count}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--state-every", type=int, default=600)
    ap.add_argument("--asset", type=Path, default=REPO / "build" / "liam.hwrm")
    ap.add_argument("--flask", default="flask_1")
    ap.add_argument("--worm", default="Liam")
    ap.add_argument("--gen", default="gen-0007")
    ap.add_argument("--tree", default="poetry-2")
    ap.add_argument("--embedder-seed", type=int, default=0)
    args = ap.parse_args()

    gen_dir = V7 / "data" / args.tree / "generations" / args.flask / args.gen / args.worm
    seed = int((gen_dir / "seed.txt").read_text().strip())

    c_bin = HERE / "hostsim" / "hostsim"
    if not c_bin.exists():
        subprocess.run(["make", "-s"], cwd=c_bin.parent, check=True)
    c_out = subprocess.run(
        [str(c_bin), str(args.asset), str(args.ticks), str(args.state_every)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()

    print(f"python: simulating {args.ticks} ticks (seed {seed})...", file=sys.stderr)
    py_out = run_python(args.ticks, args.state_every, gen_dir, seed, args.embedder_seed)

    # --- compare -------------------------------------------------------------
    py_eat = [l for l in py_out if l.startswith("EAT")]
    c_eat = [l for l in c_out if l.startswith("EAT")]
    # Compare the VALUES, not the rendered text: Python's float.hex() always pads
    # the mantissa to 13 hex digits while C's %a trims trailing zeros, so
    # 0x1.948ef659699d0p+9 and 0x1.948ef659699dp+9 are the same double printed
    # two ways. Parsing both back and comparing bit patterns is the real test.
    def parse_state(lines):
        out = {}
        for l in lines:
            if not l.startswith("STATE"):
                continue
            f = l.split()
            out[f[1]] = tuple(float.fromhex(x) for x in f[2:])
        return out

    py_state = parse_state(py_out)
    c_state = parse_state(c_out)

    def same_bits(a, b):
        import struct
        return all(struct.pack("<d", x) == struct.pack("<d", y) for x, y in zip(a, b))

    print(f"\neaten words:  python {len(py_eat)}   C {len(c_eat)}")

    first_bad = None
    for i, (p, c) in enumerate(zip(py_eat, c_eat)):
        if p != c:
            first_bad = i
            break
    if first_bad is None and len(py_eat) != len(c_eat):
        first_bad = min(len(py_eat), len(c_eat))

    if first_bad is None:
        print("  eaten-word sequence: IDENTICAL")
    else:
        print(f"  eaten-word sequence: diverges at eat #{first_bad}")
        for i in range(max(0, first_bad - 2), min(len(py_eat), first_bad + 3)):
            print(f"    py[{i}] {py_eat[i] if i < len(py_eat) else '--'}")
            print(f"    c [{i}] {c_eat[i] if i < len(c_eat) else '--'}")

    # Body state at the sampled ticks, compared bit for bit.
    exact = drifted = 0
    first_drift = None
    for k in sorted(py_state, key=int):
        if k not in c_state:
            continue
        if same_bits(py_state[k], c_state[k]):
            exact += 1
        else:
            drifted += 1
            if first_drift is None:
                first_drift = k
    print(f"\nbody state samples: {exact} bit-identical, {drifted} drifted")
    if first_drift:
        print(f"  first drift at tick {first_drift}")
        pf, cf = py_state[first_drift], c_state[first_drift]
        for name, a, b in zip(("x", "y", "facing", "speed"), pf, cf):
            d = abs(a - b)
            rel = d / max(abs(a), 1e-300)
            print(f"    {name:7s} py={a!r:<24} c={b!r:<24} |delta|={d:.3e} rel={rel:.3e}")

    ok = first_bad is None and drifted == 0
    print("\n" + ("LOCK-STEP: the board would track the server exactly."
                  if ok else
                  "DIVERGENT: same worm, its own life. See wm_math.h for why."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
