# HamletWorms

One worm from [wordswordsworms.org](https://wordswordsworms.org) living on a
Waveshare ESP32-S3-Touch-AMOLED-1.75 — a 466×466 round AMOLED — crawling over a
field of *Hamlet*, eating words, and saying each one out loud in a British voice.

No network. No SD card. Power it and the animal starts.

The simulation is the real v7 one, not an approximation: the same 300-neuron
*C. elegans* connectome, the same integrate-and-fire dynamics, the same trailing
IK body, the same learned chemosensory embedding. It runs at a true 60 Hz on the
chip.

## Why it fits in 16 MB

`v7/cache/corpus_nomic512.json` is 26 MB of frozen 512-dim word vectors, and it
never has to ship. `EmbeddingModel.prime()` projects them 512→11 with fixed
weights, and the same table serves both the current word and the eaten-word
history — so the projection precomputes to a 4919 × 11 table and everything left
is four small matmuls.

| | |
|---|---|
| worm asset — brain, corpus, taste | 1.02 MB |
| voice bank — all 4919 words | 10.47 MB |
| app | 0.36 MB |
| **total** | **11.85 MB of 16 MB** |

## Look

Black ground, white words, green animal.

True black is doing real work here: on an AMOLED those pixels are simply off, so
the ground reads as depth rather than as a dark grey panel.

**The visible circle is the worm's chemosensory horizon.** `FOOD_SENSE_RADIUS` is
200 world units, so a 400-unit window on a 466 px round display puts the rim
exactly on the edge of what the worm can smell. Words drift in from outside its
awareness and become real as they cross.

**Type is Georgia at 26 px.** Georgia was drawn for screen legibility at small
sizes; Baskerville, tried first, dissolved at this pixel density.

**Behind everything, a wireframe globe** — parallels and meridians of a sphere
under orthographic projection, tilted 23.5° so neither family collapses to a
line, with alpha falling off by depth so the far side sits behind the near side.
It is screen-fixed, not world-fixed: the camera rides the worm's head, so a
static graticule reads as the instrument you are looking through.

**Eating fires the neurons.** Nodes every eighth segment light in a wave that
travels head to tail, a pulse ring leaves the head, and the body lifts slightly
toward white — slightly, so the animal still reads as green while it happens.

The body is 800 world units long against a 400-unit window, so the tail trails
out of frame. Deliberate: the head is where the eating happens, and a microscope
does not show you the whole animal.

## Voice

Every word in the vocabulary is pre-rendered by macOS `say` in Daniel (en_GB),
silence-trimmed, speech-normalised, and packed as IMA-ADPCM at 8 kHz into one
flash blob indexed by vocabulary id — the same integer the worm asset uses, so
speaking a word the worm just ate needs no string lookup.

Loudness is fixed in the bank, not with a gain knob at playback. The rendered
words peak near half full scale but average an RMS of ~2000, a crest factor
around 8, so multiplying on the way to the codec clips the peaks off most words
without making them sound louder. `speechnorm` walks the envelope up instead,
with a limiter behind it.

Codec comparison that chose ADPCM, measured over a 59-word sample and
extrapolated to the full vocabulary:

| codec | per word | full vocabulary |
|---|---|---|
| Opus 16 kbps @16 kHz | 990 B | 4.65 MB |
| Opus 24 kbps @16 kHz | 1372 B | 6.44 MB |
| MP3 32 kbps | 2228 B | 10.45 MB |
| **IMA-ADPCM @8 kHz** | **2263 B** | **10.62 MB** |

ADPCM wins on integration risk rather than size — the decoder is about thirty
lines and costs nothing — and it already fits. `--codec opus` is wired up for
when halving it looks worth a decoder dependency.

The decoder has one trap in it. The classic IMA reference accumulates
`step>>3 + step>>2 + step>>1 + step`, truncating each term; ffmpeg computes
`((2*delta+1)*step)>>3` and truncates once. Same 1.875×step, one rounding
apart — and since ffmpeg encoded the bank, rounding the other way drifts from
the *second sample* and comes out as static. `tools/voxcheck.py` checks the
firmware's decoder against ffmpeg sample for sample.

## Layout

    tools/bake.py        v7 worm -> .hwrm asset
    tools/voices.py      macOS `say` -> IMA-ADPCM voice bank
    tools/mkfont.py      TTF -> alpha atlas + C table
    tools/crosscheck.py  diffs the C sim against Python, tick by tick
    tools/embedprobe.py  diffs the embedding forward pass, bit by bit
    tools/voxcheck.py    validates the bank and the ADPCM decoder
    tools/hostsim/       runs the sim on the host
    tools/preview/       runs sim + renderer, writes frames
    tools/flash_assets.sh

    firmware/components/wormsim/    portable C port of the v7 sim
    firmware/components/wormrender/ the rasteriser
    firmware/main/                  app, display, voice

Build the assets, then look at it without any hardware:

    python3 tools/bake.py   --out build/liam.hwrm
    python3 tools/voices.py --out build/voices.hvox
    make -C tools/preview && ./tools/preview/preview build/liam.hwrm build/anim 44000 120 4

On the board:

    cd firmware && idf.py set-target esp32s3 && idf.py build
    idf.py -p /dev/cu.usbmodem101 flash
    ../tools/flash_assets.sh          # only when the worm or the voice changes

## Performance

Measured on the board: **12–20 fps**, with the simulation holding **60.00 Hz**
exactly (36,001 ticks in 600 s). The tick rate is the number that matters — it
is the rate the server-side twin runs at too.

Four findings, none of which the host preview could have produced:

- The renderer ran in float64 like the sim and took **2.7 s a frame**. The S3's
  FPU is single-precision only, so ~400k `sqrt`/`hypot` calls a frame were
  software-emulated. Nothing in the rasteriser feeds the simulation, so nothing
  there needs float64.
- Then it took **286–475 ms** against 28 ms of transfer, because it drew into a
  PSRAM framebuffer. PSRAM is punishing for the scattered single-byte writes the
  coverage pass does. It now renders a band at a time in internal SRAM; there is
  no framebuffer at all.
- A whole-frame blit is illegal here regardless: 434 KB is over the SPI bus's
  32 KB max transaction, so esp_lcd splits it and sets
  `SPI_TRANS_CS_KEEP_ACTIVE`, which `spi_master.c` rejects unless the bus was
  acquired with `spi_device_acquire_bus()` — which esp_lcd never does. Band
  height comes from `spi_bus_get_max_transaction_len()`. The SRAM budget and the
  bus limit happen to want the same number.
- **`esp_lcd_panel_draw_bitmap` only queues the transfer.** The DMA reads the
  buffer after it returns. Handing it a band and immediately drawing the next
  one into that same buffer corrupted output silently and then wedged the task
  forever, ~100 frames in, inside `spi_device_get_trans_result` — which esp_lcd
  waits on with `portMAX_DELAY`. Two alternating band buffers, an
  `on_color_trans_done` semaphore, and a *bounded* wait fix it: a lost
  completion costs one frame, not the animal.

That last one presented as a dead serial console. What separated "the app hung"
from "the console stopped" was a heartbeat task pinned to the other core: it
kept printing while the worm task's tick and frame counters sat frozen. A stage
counter in the same log then named the exact call. Both are still in the build —
the fps line carries stack high-water marks for both tasks, dropped words, and
blit timeouts, so the next thing to go wrong says so instead of going quiet.

## Fidelity

`tools/crosscheck.py` runs the Python original and the C port on the same worm
and diffs eaten words and body state.

Bit-identical for **44,461 ticks** (~12 minutes of sim time), then it diverges.
`tools/embedprobe.py` localises the cause to **1 ULP (1.11e-16)** in the
embedding forward pass — not libm, which both sides share, but numpy: `np.exp`
is numpy's own SIMD kernel and matmul goes through BLAS with blocking and FMA,
while `wm_affine` sums left to right. The connectome fires on a hard `psyn > 30`
threshold, so one ULP eventually flips a neuron.

A generation is a full pass of the play (5498 sentences × 4.5 s spawn ≈ 6.9
hours), so drift dominates a generation. Closing it means calling the same two
kernels from both sides — see `firmware/components/wormsim/wm_math.h`, where
every operation that is not bit-exact by IEEE rule is isolated for exactly that
purpose.

Until then a board is a worm of the same lineage rather than a lock-step replica
of its server-side twin. For a wall of boards that difference is invisible; for
matching a poem on the site word for word, it is not.

## Things faithfully reproduced that look like bugs

`sim/connectome.py`'s `M_RIGHT` carries `MDL21`/`MVL21` where `MDR21`/`MVR21`
belong — a typo from the GoPiGo original that worm-sim kept and v7 kept after
it. So those two muscles are visited twice and credited to the left both times,
and `MDR21`/`MVR21` are never read or cleared and accumulate charge forever. The
C port does the same. Fixing it would change the animal.

The eaten-word history is not removable either. In v7 the worm's last five eaten
words *are* its memory: they feed the embedding, so the same word on screen
tastes different depending on what went before it. Drop them and it is a
different creature.
