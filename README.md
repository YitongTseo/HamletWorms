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
| app | 1.12 MB |
| **total** | **12.61 MB of 16 MB** |

The app was 0.36 MB before the genome sync went in; most of what it gained since
is the WiFi stack. 31 KB of it is the type going from 34 px to 42 — the glyph
atlas is uncompressed 8-bit alpha, so its cost is quadratic in the point size.

## Look

Black ground, black words in white clouds, green animal, and one dark thing
turning behind all three.

True black is still doing real work: on an AMOLED those pixels are simply off,
so the ground reads as depth rather than as a dark grey panel. Everything added
since is kept dim enough not to spend that — which is also why the words went
from white to black. White type on a white haze is the worst of both; the haze
had to be either dark enough to leave the type alone, which is no haze at all,
or bright enough to be ink on paper. It is ink on paper.

**The visible circle is the worm's chemosensory horizon.** `FOOD_SENSE_RADIUS` is
200 world units, so a 400-unit window on a 466 px round display puts the rim
exactly on the edge of what the worm can smell. Words drift in from outside its
awareness and become real as they cross.

**Type is Georgia at 42 px.** Georgia was drawn for screen legibility; Baskerville,
tried first, dissolved at this pixel density. 42 overlaps neighbouring words in a
crowded line, and that is the trade taken — a word you can read from across the
room is worth more than a line that never collides.

**The camera lags the head by 2.5 seconds, on a 105 px leash.** It used to sit
exactly on it, and that turned out to be why nothing else in this section
worked. The head bobs at a mean 44 px/s and peaks at 127; words drift up at
17.5. So a word's motion on screen was mostly head-twitch — it crossed a fade
band in a fraction of a second and then crossed straight back, and no width of
band could have fixed it. Low-passing the camera takes that to a mean 23 px/s.

Lag alone is not enough, because a good part of the head's motion is the animal
genuinely travelling and the camera has to go too: at 4 s of lag the head
wanders 256 px off centre and leaves the frame. So the filter is long and the
offset is clamped — follow slowly, drag the rest of the way past 105 px. Smooth
where it can be, hard where it must be. The animal now weaves inside the frame
instead of being pinned to the middle of it, which is also what makes the fixed
dot lattice read as something it is passing under.

**Words arrive rather than appear.** A word's weight is its distance from the
middle of the panel: full inside 60% of the radius, nothing at the bezel. That
is 103 px of travel, about four seconds, and it is the part that does the work —
the panel is a circle, so a word off to one side crosses the bezel on an arc
rather than through the bottom, and a purely vertical profile let it switch on
the moment it cleared the mask. It is also the peripheral-vision reading the
framing already claims: what is off to the side is hazy, what is in the middle
is sharp.

On top of that, a vertical profile: over the bottom of the panel a word grows
from 54% size and fades up, holds through the middle, then over the top third
fades out and draws back down, which reads as passing overhead rather than as
being deleted.

Growth has to be continuous, so a scaled word resamples the atlas bilinearly
rather than picking a smaller baked size: stepping between two baked sizes moves
a word's edges several pixels at once, which is visible as a twitch even under a
fade. Full size takes the straight blit, and that is where words spend most of
their life.

**Each live word is black, sitting in a cloud, casting a shadow onto it.** The
cloud comes first: it is the only reason black type is legible on a black panel
at all, and the only reason a shadow has anywhere to land. So it is flat-topped
rather than peaked — a profile that falls off from the centre leaves the ends of
a long word on a dimmer ground than its middle, which black type shows
immediately — and it outlives the word slightly, fading as `alpha^0.7`, so there
is always something to be black against at both ends of the word's life.

The shadow is violet-grey, not black. The word itself is nearly black now, so a
black shadow under it is just a heavier letter; what should read is the sliver
that falls past the stroke onto the cloud.

Set-dressing (speaker names, stage cues) and already-eaten words get no cloud,
so they stay light — pale blue and spent green. Only the food is ink.

**Behind everything, an alien sphere.** Three-dimensional value noise on the
surface of an orthographic ball, ridged so the level sets are crests rather than
plateaux, with the noise's own domain warped by a coarser octave of itself — so
the pattern is dragged through itself as the ball turns and the thing folds
inward instead of merely drifting. A fixed key light and a cold fresnel at the
limb make it read as a solid rather than as a disc of texture. It is violet, not
green: the animal is the only green thing on the panel and has to stay that way.

The world-anchored wireframe globe is still there at about a fifth of its old
weight. It is the only thing on the panel that says where in the play the worm
currently is — its poles sit on `SPAWN_Y` and `KILL_Y` — but it is no longer the
background, and `wr_ctx.globe_alpha` will take it to zero.

**The animal is made of dots.** A clustered-dot threshold screen, tiled and
fixed to the panel: a pixel lights when the body's coverage there exceeds its
threshold, and the threshold rises as the square of the distance to the nearest
dot centre, so a dot's area is linear in coverage. The lattice does not move, so
the worm reads as something passing under a fabric rather than as something
wearing a texture. The gain is held under full scale on purpose — at 255 the
dots flood together into the old solid body and there is nothing to see through.

Coverage runs past the silhouette and falls to zero over 9 px, so the dots shrink
outward into a fringe, and the same screen at its lowest threshold puts a sparse
dot of dust across the whole panel. That is the reading: a field of dots, and the
animal is where they thicken.

**The animal darkens what it crosses.** Its own coverage pushes the ground under
it down by up to 72% before any of its dots are laid on top. Without that, green
dots landing on a white word-cloud sit at nearly the same value as the black
type beside them and the two read as one texture; with it the worm always has a
dark ground of its own and is unmistakably in front. The words stay legible
through it, which is the point — it is a shadow, not an eraser.

The rainbow is gone. It was never a pattern — it was per-capsule quantisation of
a gradient that ran head colour into body colour over the front 18% of the
animal, steep enough that each capsule landed on a visibly different shade. One
hue now, with the banding put back deliberately as a slow brightness wave: 13
bands at ±14%, which is roughly what the body-wall muscles actually do.

**Eating fires the neurons.** Nodes every eighth segment light in a wave that
travels head to tail, a pulse ring leaves the head, and the body lifts slightly
toward white — slightly, so the animal still reads as green while it happens.

The body is 800 world units long against a 400-unit window, so the tail trails
out of frame. Deliberate: the head is where the eating happens, and a microscope
does not show you the whole animal.

### Everything low-frequency happens on a 64x64 grid

The ball and the clouds behind the words are both genuinely low-frequency —
nothing in either has detail finer than seven pixels. So neither is evaluated per
pixel. They share one 64x64 RGB field: the ball is written into it (every third
frame — it turns at 0.11 rad/s, and there is nothing in it that moves far enough
in three frames to see), each live word adds an elliptical bloom to a copy, and
the whole thing is bilinearly resampled on the way to the panel.

That is 4096 samples and a resample instead of 217156 shader evaluations, and it
is the only reason any of this fits in the frame budget. The resample itself is
the most expensive per-pixel loop in the renderer, so it does the horizontal
half once per source row into RGB565 and leaves one packed blend per pixel; it
skips the fifth of every band that lies outside the round panel; and the dust
comes from a precomputed list of positions rather than a per-pixel test.

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

Measured on the board: **6.3–7.6 fps**, with the simulation holding **60 Hz**
exactly (3633 ticks in 60.7 s). The tick rate is the number that matters — it is
the rate the server-side twin runs at too, and none of what follows touches it.

The look before the sphere, the clouds and the stipple ran at **11.3–13.7 fps**
on the same board. The new renderer costs **1.9×** the draw time, 52–64 ms
becoming 104–124 ms:

| stage | ms |
|---|---|
| the ball, plus the frame's word list and its haze | 11–13 |
| resampling the background onto the panel | 21 |
| graticule | 6–10 |
| words | 7–14 |
| stippled body, incl. the ground it lays down | 50–65 |

The host preview predicted 1.7× and was right, which is worth recording because
the findings below are all cases of it being wrong. What it got wrong here is
the *shape*: the background resample is 210× slower on the S3 than on the host,
where the stippled body is only 114× slower. A tight integer loop over 172,000
pixels is what this chip is worst at relative to a desktop, and it is the thing
to attack next.

Getting even to 1.8× took three cuts. Two of them show up as host stage totals:

| | before | after |
|---|---|---|
| background resample | 0.48 ms | 0.10 ms |
| stippled body incl. fringe | 0.80 ms | 0.44 ms |

The background field first interpolated in 8-bit RGB and packed to 565 per
pixel. Doing the horizontal half once per source row and storing *that* in 565
leaves one packed blend per output pixel — and it costs no fidelity, because the
panel is 565 either way, so both paths land on the same colours.

The fringe outside the body was a 9 px annulus rasterised around all 200 body
capsules. A 9 px annulus around a 5 px segment is almost entirely bounding box:
it cost more than the animal did. It is now a disc stamped every fourth segment,
walked by exact per-row spans rather than a box, which leaves the envelope
scalloped by about two pixels — nothing rendered as dots can show that.

And the ball is rebuilt every third frame rather than every frame.

Everything added is behind a runtime switch, so a board that turns out not to
afford it can give any of it back: `wr_ctx.bg_alien`, `haze`, `stipple`,
`globe_alpha`. Turning the ball off is worth about 32 ms a frame on its own,
which is most of the way back to the old frame rate.

Five findings, none of which the host preview could have produced:

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

- **24 KB of internal DRAM was the difference between an animal and a
  slideshow.** The renderer's background field was allocated
  `MALLOC_CAP_INTERNAL` first, like everything else here. It fit — and then
  `xTaskCreatePinnedToCore` for the worm task could not find 8 KB for its stack,
  returned `pdFAIL`, and nothing checked it. The board booted, logged a clean
  startup, printed a heartbeat every second and sat on a frozen tick count
  forever: the 3 KB heartbeat had fit where the 8 KB worm did not. The field is
  PSRAM now, which is where it belonged — it is written 4096 cells every third
  frame and read back a 192-byte row at a time, nothing like the scattered
  single-byte writes that made PSRAM unusable for the band scratch. Both
  `xTaskCreate` calls are checked now, and the free internal heap is logged just
  before them.

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
