# HamletWorms

One worm from [wordswordsworms.org](https://wordswordsworms.org) living on a
Waveshare ESP32-S3-Touch-AMOLED-1.75 — a 466x466 round AMOLED — crawling over a
field of Hamlet, eating words, and saying each one out loud.

The sim is the real v7 simulation, not an approximation: the same 300-neuron
connectome, the same integrate-and-fire dynamics, the same trailing IK body, the
same learned chemosensory embedding. It runs standalone on the chip with no
network.

## Why it fits

`cache/corpus_nomic512.json` in v7 is 26 MB of frozen 512-dim word vectors, and
it never has to ship. `EmbeddingModel.prime()` projects them 512->11 with fixed
weights, and the same table serves both the current word and the eaten-word
history — so the projection precomputes to a 4919 x 11 table and everything else
is four small matmuls.

| | |
|---|---|
| worm asset (brain, corpus, taste) | 1.02 MB |
| voice bank, all 4919 words | 10.47 MB |
| **total flash payload** | **11.49 MB** of 16 MB |

## Layout

    tools/bake.py        v7 worm -> .hwrm asset
    tools/voices.py      macOS `say` -> IMA-ADPCM voice bank
    tools/mkfont.py      TTF -> alpha atlas + C table (Baskerville)
    tools/crosscheck.py  diffs the C sim against Python, tick by tick
    tools/embedprobe.py  diffs the embedding forward pass, bit by bit
    tools/voxcheck.py    validates the bank and the ADPCM decoder
    tools/hostsim/       runs the sim on the host
    tools/preview/       runs sim + renderer, writes frames

    firmware/components/wormsim/    portable C port of the v7 sim
    firmware/components/wormrender/ rasteriser for the round display

Build the assets, then look at it:

    python3 tools/bake.py --out build/liam.hwrm
    python3 tools/voices.py --out build/voices.hvox
    make -C tools/preview && ./tools/preview/preview build/liam.hwrm build/anim 44000 120 4

## Fidelity

`tools/crosscheck.py` runs the Python original and the C port on the same worm
and diffs eaten words and body state.

Currently bit-identical for **44,461 ticks** (~12 minutes of sim time), then it
diverges. `tools/embedprobe.py` localises the cause to **1 ULP (1.11e-16)** in
the embedding forward pass — not libm, which both sides share, but numpy:
`np.exp` is numpy's own SIMD kernel and matmul goes through BLAS with blocking
and FMA, while `wm_affine` sums left to right. The connectome fires on a hard
`psyn > 30` threshold, so one ULP eventually flips a neuron.

A generation is a full pass of the play (5498 sentences x 4.5 s spawn ~= 6.9
hours), so drift dominates a generation. Closing it means calling the same two
kernels from both sides — see `firmware/components/wormsim/wm_math.h`, where
every non-IEEE-exact operation is isolated for exactly that purpose.

Until then a board is a worm of the same lineage rather than a lock-step replica
of its server-side twin.

## Faithfully reproduced bugs

`sim/connectome.py`'s `M_RIGHT` carries `MDL21`/`MVL21` where `MDR21`/`MVR21`
belong — a typo from the GoPiGo original that worm-sim kept and v7 kept after
it. So those two muscles are visited twice and credited to the left both times,
and `MDR21`/`MVR21` are never read or cleared and accumulate charge forever.
The C port does the same. Fixing it would change the animal.
