// wormrender — draws one worm and its word field, band by band.
//
// Portable C, no ESP-IDF: tools/preview builds it on the host and writes PNGs,
// so the look can be settled before touching hardware.
//
// Framing: the visible circle is the worm's chemosensory horizon.
// FOOD_SENSE_RADIUS is 200 world units, so a 400-unit window mapped onto the
// 466 px round display puts the edge of the screen exactly on the edge of what
// the worm can smell. Words drift in from outside its awareness and become real
// as they cross the rim.
//
// The body is 200 segments x 4.0 units = 800 units long, twice the window, so
// the tail trails out of frame. That is deliberate — the head is where the
// eating happens, and a microscope does not show you the whole animal.
//
// Output goes out a band at a time through a caller-supplied blit rather than
// into a framebuffer. On the board that keeps the per-pixel work in internal
// SRAM and lets each band DMA straight to the panel; a 466x466 PSRAM
// framebuffer cost 286-475 ms a frame against 28 ms of actual transfer.

#ifndef WORMRENDER_H
#define WORMRENDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "wormsim.h"

#define WR_W 466
#define WR_H 466
#define WR_N_GLYPHS 96  // printable ASCII 32..126, plus the em dash at slot 95
#define WR_GLYPH_EM_DASH 95

// Half-width of a graticule line, px. Thin enough to stay behind the words.
#define WR_FRAME_HALF_WIDTH 1.15f

// The background is computed on a coarse grid and bilinearly resampled up to
// the panel. Everything in it — the alien sphere and the haze behind the words —
// is low-frequency by design, so 64x64 carries all the detail there is and the
// per-pixel cost collapses from a shader to two lerps.
#define WR_BG_N 64

// Screen-space stipple screen, tiled. A clustered-dot threshold field: a pixel
// lights when the body's coverage there exceeds its threshold, so dots grow out
// of their centres as the animal arrives under them.
#define WR_STIPPLE_TILE 64

typedef struct {
    uint16_t ax, ay;  // position in the alpha atlas
    uint8_t w, h;
    int8_t bx, by;    // bearing from the pen position on the baseline
    uint8_t adv;
} wr_glyph;

extern const wr_glyph wr_font_glyph[WR_N_GLYPHS];
extern const uint8_t wr_font_atlas[];
extern const int wr_font_atlas_w, wr_font_atlas_h;
extern const int wr_font_size, wr_font_ascent, wr_font_descent;

// Palette. Black ground, white words, green animal — the specimen-under-glass
// reading, and the highest contrast the panel can give on all three.
#define WR_STAGE 0x000000   // true black; an AMOLED pixel here is simply off
#define WR_WORD 0x0D0B14        // ink: an edible word is black, in its own cloud
#define WR_WORD_CAST 0x2E2340   // the shadow it throws onto that cloud
#define WR_WORD_INERT 0x8FA6C4  // set-dressing: names, stage cues
#define WR_GHOST 0x2E6B4E       // a word already swallowed
#define WR_ACCENT 0x3DDC84      // the worm
#define WR_HEAD 0x9CF7C4        // its anterior, catching the light
#define WR_FIRE 0xEAFFF2        // neurons, mid-flash
#define WR_GUT 0x1B5B3C         // the intestine, seen through the cuticle
#define WR_PHARYNX 0x71D6A8     // the pharynx, denser than what surrounds it
#define WR_GLOBE 0xC4CCD4       // the graticule: cool grey, deliberately not green
#define WR_HAZE 0xF2F6FF        // the cloud a word sits in
#define WR_DUST 0x2A4A5E        // the dot field the worm surfaces through

// The sphere behind everything, dark to bright. Violet rather than green: the
// animal is the only green thing on the panel and has to stay that way.
#define WR_BG_LOW 0x030208
#define WR_BG_MID 0x120A2A
#define WR_BG_HIGH 0x4E2C78
#define WR_BG_RIM 0x1C4658

typedef struct {
    // Two band buffers, alternated. The panel's DMA reads a band after
    // draw_bitmap has already returned, so with a single buffer the renderer
    // would be overwriting pixels that are still being transmitted. Alternating
    // lets band N transmit while band N+1 is drawn.
    uint16_t *band;      // the one being drawn into
    uint16_t *band_mem;  // both, back to back
    int band_parity;
    uint8_t *cov;    // per-pixel body coverage
    uint8_t *seg;    // per-pixel position along the body, for the gradient
    int band_rows;

    // World units across the display. 400 puts the rim on the smell radius.
    float view_units;

    // Camera centre, world units — an output, read back by tools. It follows
    // the worm's head through a low-pass of cam_lag seconds.
    float cam_x, cam_y;

    // Seconds of lag between the head and the camera. The head bobs at a mean
    // 44 px/s and peaks at 127, against words that drift up at 17.5, so a camera
    // welded to it makes the word field's screen motion mostly head-twitch —
    // and any fade driven by screen position gets crossed in a fraction of a
    // second and then crossed back. Following the head's average instead lets a
    // word take the ten seconds it should to arrive, and lets the animal weave
    // inside the frame instead of being pinned to the middle of it.
    //
    // Zero welds it back to the head. Driven off bg_time, so it is frame-rate
    // free; if bg_time never advances this degrades to welded rather than to a
    // camera that never moves.
    float cam_lag;
    float cam_sx, cam_sy, cam_t;  // filter state
    bool cam_have;

    // Body half-width in world units. The site uses 22 for a camera that sees
    // the whole animal; at this magnification that reads as a snake.
    float body_radius;

    // 1.0 the instant a word is eaten, decaying after. Drives the neuron flash
    // along the body and the pulse ring off the head.
    float flash;

    // Identity card, shown briefly at startup. With several boards on a wall
    // there is otherwise no way to tell which worm you are looking at.
    const char *title;     // the worm's name
    const char *subtitle;  // flask and generation
    float title_alpha;     // 0..1, driven by the caller

    // Held while a finger is down. A straight XOR of the framebuffer, so the
    // ground goes white, the words black and the animal magenta — unmistakable,
    // and one instruction per pixel.
    // Held while a finger is down. A straight XOR of the framebuffer. Left in
    // and wired, but off by default now that touch shows the x-ray instead —
    // the two fight for the same pixels.
    bool invert;

    // Held while a finger is down: dim the body and light the connectome
    // inside it, the way viewer/focus/xray-render.js does.
    bool xray;

    // Decays after a poke. Jitters the camera, which reads as the worm
    // flinching without touching the simulation's own state.
    float shudder;
    uint32_t frame;  // drives the jitter; bumped by wr_draw_banded

    bool round_mask;  // zero the corners outside the circular panel

    // How much of the world-anchored graticule survives, 0..1. The alien sphere
    // took over the job of being the background; the wireframe stays because it
    // is the only thing on the panel that says where in the play the worm is,
    // but at a fraction of its old weight.
    float globe_alpha;

    // The sphere: value noise on the surface of an orthographic ball, its
    // domain warped by another octave of itself so it folds inward as it turns.
    bool bg_alien;
    float bg_time;  // seconds; advanced by the caller so it is frame-rate free

    // Haze behind the words, 0..1. A word carries a soft white cloud in the
    // background field, which is what lets a black drop shadow read at all.
    float haze;

    // 0 = a solid body, 1 = the animal entirely resolved into dots.
    float stipple;

    // Draw what a nematode actually has inside it: the pharynx behind the
    // mouth, and the gut running back from it. Cheap — two dozen discs along a
    // midline that is already projected — and it is the difference between a
    // green tube and something that reads as an animal.
    bool anatomy;

    uint32_t cov_pixels;  // body pixels drawn last frame (diagnostic)
} wr_ctx;

// Called once per band with rows [y, y + h) of the frame.
//
// May return before the pixels have actually been sent. The renderer will not
// touch that buffer again until it has called blit once more with the other
// one, so an implementation that transmits asynchronously must wait for the
// PREVIOUS transfer to finish before starting this one, and the caller must
// drain the last transfer after wr_draw_banded returns.
typedef void (*wr_blit_fn)(void *user, int y, int h, const uint16_t *pixels);

// Scratch for one band: pixels + coverage + body position. Put it in internal
// SRAM — that is the entire point of banding.
// Per-stage timing in microseconds, accumulated across bands. Point wr_clock at
// a microsecond source to enable.
extern uint32_t wr_us_sphere, wr_us_bg, wr_us_frame, wr_us_words, wr_us_worm;
extern uint32_t (*wr_clock)(void);

size_t wr_scratch_bytes(int band_rows);

// The coarse background: the ball, and a working copy of it with this frame's
// haze added. Two WR_BG_N x WR_BG_N RGB grids — 24 KB, written a few thousand
// cells at a time and read a row at a time, so PSRAM is fine if internal SRAM is
// wanted elsewhere. Unlike the band scratch, nothing here is handed to the DMA.
size_t wr_bg_bytes(void);

// Precomputes the graticule's world coordinates. Once, at startup.
void wr_build_globe(void);

void wr_init(wr_ctx *c, uint8_t *scratch, int band_rows, uint8_t *bg);
void wr_draw_banded(wr_ctx *c, const wm_world *w, wr_blit_fn blit, void *user);

#endif  // WORMRENDER_H
