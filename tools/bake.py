#!/usr/bin/env python3
"""Bake one HamletRNAWorld v7 worm into a single flat asset file the firmware mmaps.

The point of this tool is that almost nothing about v7 actually has to run on the
ESP32. In particular `cache/corpus_nomic512.json` is 26 MB of frozen 512-dim
vectors — but look at `EmbeddingModel.prime()`: the 512->11 projection E is
applied with fixed weights, and the *same* table serves both the current word and
the eaten-word history. So E is precomputable per word, and the 26 MB collapses to
a 4919 x 11 float table. What ships is ~600 KB total.

What still runs on-chip: the connectome integrate-and-fire, the IK chain, the text
scroller, and four tiny matmuls (Hh 55->11, Hf 22->11, P1 72->16, P2 16->1).

Orderings in here are load-bearing. Float addition is not associative, so the
firmware has to accumulate synapses in exactly the order CPython's dict iteration
produces, and iterate neurons in exactly `sorted()` order. Every table below is
emitted in the order the Python reads it, never re-sorted for convenience.

Usage:
    python3 tools/bake.py --out build/liam.hwrm
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

# --- locate the v7 tree and make its packages importable ---------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V7 = REPO.parent / "HamletRNAWorld" / "v7"

if not V7.exists():
    sys.exit(f"cannot find v7 tree at {V7}")
sys.path.insert(0, str(V7))

# pos_scorers silently returns "X" for every word when the NLTK tagger data is
# missing, which would quietly flatten PC11 to a constant. Point NLTK at the
# copy we vendored under tools/ before importing anything that uses it.
os.environ.setdefault("NLTK_DATA", str(HERE / "nltk_data"))

import numpy as np  # noqa: E402

from corpus.hamlet import get_sentences_with_flags  # noqa: E402
from server import embedding, pos_scorers  # noqa: E402
from sim import connectome as conn  # noqa: E402
from sim.chemosensory_mapping import PC_NEURON_PAIRS  # noqa: E402

MAGIC = b"HWRM"
VERSION = 1


# --- section writer ----------------------------------------------------------
class Blob:
    """Accumulates little-endian fields. ESP32-S3 and x86 are both LE, so the
    firmware can point structs straight at the mapped bytes."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def u8(self, v):
        self.buf += struct.pack("<B", v)

    def u16(self, v):
        self.buf += struct.pack("<H", v)

    def u32(self, v):
        self.buf += struct.pack("<I", v)

    def f32(self, v):
        self.buf += struct.pack("<f", v)

    def raw(self, b):
        self.buf += b

    def arr_u8(self, vals):
        self.buf += np.asarray(vals, dtype="<u1").tobytes()

    def arr_u16(self, vals):
        self.buf += np.asarray(vals, dtype="<u2").tobytes()

    def arr_u32(self, vals):
        self.buf += np.asarray(vals, dtype="<u4").tobytes()

    def arr_f64(self, vals):
        # float64, not float32, everywhere a value feeds arithmetic. Python and
        # numpy are float64 throughout; the connectome's fire test is a hard
        # `psyn > 30` threshold, so truncating a synaptic weight to float32 flips
        # a neuron within seconds and the board walks away from its server twin.
        # 433 KB for the E-table is a rounding error against 16 MB of flash.
        self.buf += np.asarray(vals, dtype="<f8").ravel().tobytes()

    def align(self, n=8):
        while len(self.buf) % n:
            self.buf += b"\0"

    def strtab(self, strings):
        """u32 count, u32 blob_len, u32 off[count+1], then the utf-8 blob."""
        enc = [s.encode("utf-8") for s in strings]
        offs, cur = [], 0
        for e in enc:
            offs.append(cur)
            cur += len(e)
        offs.append(cur)
        self.u32(len(enc))
        self.u32(cur)
        self.arr_u32(offs)
        self.raw(b"".join(enc))
        self.align(4)


def pack(sections: list[tuple[str, bytes]]) -> bytes:
    """Header + TOC + 16-byte-aligned section payloads."""
    head = bytearray()
    head += MAGIC
    head += struct.pack("<III", VERSION, len(sections), 0)
    toc_size = 16 * len(sections)
    body_start = (len(head) + toc_size + 15) // 16 * 16

    toc = bytearray()
    payload = bytearray()
    cur = body_start
    for tag, data in sections:
        assert len(tag) <= 8, tag
        toc += tag.encode("ascii").ljust(8, b" ")
        toc += struct.pack("<II", cur, len(data))
        payload += data
        pad = (-len(data)) % 16
        payload += b"\0" * pad
        cur += len(data) + pad

    out = bytearray(head + toc)
    out += b"\0" * (body_start - len(out))
    out += payload
    return bytes(out)


# --- the bake ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flask", default="flask_1")
    ap.add_argument("--worm", default="Liam")
    ap.add_argument("--gen", default="gen-0007")
    ap.add_argument("--tree", default="poetry-2")
    ap.add_argument("--passage", default="full")
    ap.add_argument(
        "--embedder-seed",
        type=int,
        default=0,
        help="PLACEHOLDER. The real evolved genome lives in data/poetry_shared/ "
        "on the running servers and is gitignored, so it is not in the repo yet. "
        "random_init(seed) is what get_model() cold-starts with, so this is at "
        "least the same distribution and fully deterministic. Swap in the real "
        "genome later with --embedder-genome; no firmware change needed.",
    )
    ap.add_argument(
        "--embedder-genome",
        type=Path,
        default=None,
        help="path to a flat .npy/.json genome vector to use instead of the seed",
    )
    ap.add_argument("--out", type=Path, default=REPO / "build" / "worm.hwrm")
    args = ap.parse_args()

    gen_dir = V7 / "data" / args.tree / "generations" / args.flask / args.gen / args.worm
    if not gen_dir.exists():
        sys.exit(f"no such worm: {gen_dir}")

    seed = int((gen_dir / "seed.txt").read_text().strip())
    fitness = json.loads((gen_dir / "fitness.json").read_text())
    weights = json.loads((gen_dir / "weights.json").read_text())  # insertion order preserved

    print(f"worm    {args.flask}/{args.worm} {args.gen}  seed={seed}  fitness={fitness['fitness']:.3f}")

    # -- embedder params ------------------------------------------------------
    if args.embedder_genome:
        raw = args.embedder_genome
        vec = np.load(raw) if raw.suffix == ".npy" else np.asarray(json.loads(raw.read_text()))
        params = embedding.EmbeddingParams.from_flat(vec)
        emb_src = str(raw)
    else:
        params = embedding.EmbeddingParams.random_init(args.embedder_seed)
        emb_src = f"PLACEHOLDER random_init(seed={args.embedder_seed})"
    model = embedding.EmbeddingModel(params)
    print(f"embedder {emb_src}")

    # -- vocab + E-table ------------------------------------------------------
    # model._words is `list(self._nomic.keys())`, i.e. the on-disk `words` order,
    # and _widx maps into it. The firmware indexes with the same integers.
    vocab = model._words
    etable = np.asarray(model._E_table, dtype=np.float64)
    assert etable.shape == (len(vocab), embedding.D_EMB), etable.shape
    print(f"vocab   {len(vocab)} words, E-table {etable.shape} = {etable.nbytes/1024:.0f} KB fp64")

    # -- connectome -----------------------------------------------------------
    # Mirror Connectome.__init__ exactly: every pre- and post-synaptic name, plus
    # every muscle even if it never receives a synapse, then sorted().
    names = set(weights.keys())
    for tgt in weights.values():
        names.update(tgt.keys())
    names.update(conn.MUSCLE_LIST)
    neurons = sorted(names)
    nidx = {n: i for i, n in enumerate(neurons)}

    row_start, cols, ws = [0], [], []
    for n in neurons:
        # dendrite_accumulate walks `self.weights[pre].items()` — JSON insertion
        # order. Preserve it verbatim: float addition is not associative.
        for post, w in weights.get(n, {}).items():
            cols.append(nidx[post])
            ws.append(w)
        row_start.append(len(cols))
    print(f"brain   {len(neurons)} neurons, {len(cols)} edges")

    # motorcontrol() walks MUSCLE_LIST = M_LEFT + M_RIGHT *with its duplicates*.
    # M_RIGHT carries "MDL21"/"MVL21" where you would expect MDR21/MVR21 — a typo
    # inherited from the GoPiGo original that worm-sim kept and v7 kept after it.
    # Consequences, both of which the firmware must reproduce:
    #   - MDL21/MVL21 are visited twice; `if m in _left_set` wins both times, so
    #     the second visit adds the already-zeroed 0.0 to accum_LEFT, not right.
    #   - MDR21/MVR21 never appear, so their psyn buckets are never read or
    #     cleared and just accumulate charge forever.
    m_left, m_right = set(conn.M_LEFT), set(conn.M_RIGHT)
    mus_idx, mus_side = [], []
    for m in conn.MUSCLE_LIST:
        mus_idx.append(nidx[m])
        mus_side.append(0 if m in m_left else (1 if m in m_right else 2))
    dupes = [m for m in conn.MUSCLE_LIST if conn.MUSCLE_LIST.count(m) > 1]
    print(f"muscles {len(mus_idx)} visits, duplicated: {sorted(set(dupes))}")

    # run() skips a neuron when name[:3] is a muscle prefix — precompute the test.
    is_muscle = [1 if n[:3] in conn.MUSCLE_PREFIXES else 0 for n in neurons]

    # -- corpus ---------------------------------------------------------------
    sentences, edible = get_sentences_with_flags(args.passage)
    widx = model._widx
    tok_vocab, tok_pos, tok_text, sent_start = [], [], [], [0]
    for s in sentences:
        for t in s:
            tok_vocab.append(widx.get(embedding._norm(t), 0xFFFF))
            # tag_word lowercases but does NOT strip apostrophes, unlike _norm —
            # so "'tis" tags as "'tis" while embedding-looking-up as "tis".
            # Tag the surface token, per word, to match embed_batch().
            tok_pos.append(embedding._POS_INDEX.get(pos_scorers.tag_word(t), embedding._POS_INDEX["X"]))
            tok_text.append(t)
        sent_start.append(len(tok_vocab))
    oov = sum(1 for v in tok_vocab if v == 0xFFFF)
    print(
        f"corpus  {len(sentences)} sentences, {len(tok_vocab)} tokens, "
        f"{oov} OOV ({100*oov/len(tok_vocab):.1f}%), {sum(edible)} edible sentences"
    )

    # -- sections -------------------------------------------------------------
    S: list[tuple[str, bytes]] = []

    meta = {
        "flask": args.flask, "worm": args.worm, "gen": args.gen, "seed": seed,
        "fitness": fitness["fitness"], "embedder": emb_src, "passage": args.passage,
        "n_vocab": len(vocab), "n_neurons": len(neurons), "n_edges": len(cols),
        "n_sentences": len(sentences), "n_tokens": len(tok_vocab),
    }
    b = Blob(); b.strtab([json.dumps(meta, indent=2)]); S.append(("META", bytes(b.buf)))

    b = Blob(); b.strtab(vocab); S.append(("VOCAB", bytes(b.buf)))

    b = Blob(); b.u32(len(vocab)); b.u32(embedding.D_EMB); b.arr_f64(etable)
    S.append(("ETABLE", bytes(b.buf)))

    # Fixed order; the firmware's struct mirrors it field for field.
    b = Blob()
    for name in ("W_Hh", "b_Hh", "W_Hf", "b_Hf", "W_P1", "b_P1", "W_P2", "b_P2"):
        b.arr_f64(getattr(params, name))
    S.append(("NET", bytes(b.buf)))

    b = Blob(); b.strtab(neurons); b.arr_u8(is_muscle); b.align(4)
    S.append(("NEURONS", bytes(b.buf)))

    b = Blob()
    b.u32(len(neurons)); b.u32(len(cols))
    b.arr_u32(row_start); b.arr_u16(cols); b.align(8); b.arr_f64(ws)
    S.append(("SYNAPSE", bytes(b.buf)))

    b = Blob()
    b.u32(len(mus_idx)); b.arr_u16(mus_idx); b.arr_u8(mus_side); b.align(4)
    S.append(("MUSCLES", bytes(b.buf)))

    # rand_excite does `self.rng.choice(list(self.weights.keys()))` — the JSON
    # key order, NOT the sorted neuron list. Different list, different length
    # (300 vs 396), so a different number of rejection-sampling draws. Getting
    # this wrong desynchronises the RNG stream for the entire run.
    presyn = [nidx[n] for n in weights.keys()]
    b = Blob(); b.u32(len(presyn)); b.arr_u16(presyn); b.align(4)
    S.append(("PRESYN", bytes(b.buf)))

    b = Blob()
    for L, R in PC_NEURON_PAIRS:
        b.u16(nidx[L]); b.u16(nidx[R])
    S.append(("CHEMOMAP", bytes(b.buf)))

    b = Blob()
    for group in (conn.HUNGER_NEURONS, conn.NOSE_TOUCH_NEURONS, conn.FOOD_SENSE_NEURONS):
        b.u32(len(group)); b.arr_u16([nidx[n] for n in group])
    b.align(4)
    S.append(("STIMSETS", bytes(b.buf)))

    b = Blob()
    b.u32(len(sentences)); b.u32(len(tok_vocab))
    b.arr_u32(sent_start)
    b.arr_u16(tok_vocab)
    b.arr_u8(tok_pos)
    # Layout uses len(token) in CODE POINTS. _TOKEN_RE emits an em dash, which is
    # one character to Python but three bytes in UTF-8 — counting bytes in C
    # would shift the whole line and change which words the worm can reach.
    b.arr_u8([len(t) for t in tok_text])
    b.arr_u8([1 if e else 0 for e in edible])
    b.align(8)
    b.strtab(tok_text)          # display text, original capitalisation
    S.append(("CORPUS", bytes(b.buf)))

    b = Blob(); b.u32(seed); S.append(("SEED", bytes(b.buf)))

    out = pack(S)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(out)

    print()
    for tag, data in S:
        print(f"  {tag:<9} {len(data):>9,} B")
    print(f"  {'TOTAL':<9} {len(out):>9,} B  ({len(out)/1024/1024:.2f} MB) -> {args.out}")


if __name__ == "__main__":
    main()
