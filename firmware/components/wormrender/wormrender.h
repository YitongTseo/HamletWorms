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
#define WR_WORD 0xFFFFFF
#define WR_WORD_INERT 0x8FA6C4  // set-dressing: names, stage cues
#define WR_GHOST 0x2E6B4E       // a word already swallowed
#define WR_ACCENT 0x3DDC84      // the worm
#define WR_HEAD 0x9CF7C4        // its anterior, catching the light
#define WR_FIRE 0xEAFFF2        // neurons, mid-flash
#define WR_GLOBE 0x1E6B4C       // the graticule behind everything

typedef struct {
    uint16_t *band;  // WR_W x band_rows, RGB565
    uint8_t *cov;    // per-pixel body coverage
    uint8_t *seg;    // per-pixel position along the body, for the gradient
    int band_rows;

    // Full-screen alpha for the globe graticule, built once by wr_build_globe.
    // Screen-fixed rather than world-fixed: the camera rides the worm's head,
    // so a static graticule reads as the instrument you are looking through
    // rather than as scenery the worm crawls over.
    const uint8_t *globe;

    // World units across the display. 400 puts the rim on the smell radius.
    float view_units;

    // Camera centre, world units. Normally the worm's head.
    float cam_x, cam_y;

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

    bool round_mask;  // zero the corners outside the circular panel

    uint32_t cov_pixels;  // body pixels drawn last frame (diagnostic)
} wr_ctx;

// Called once per band with rows [y, y + h) of the frame.
typedef void (*wr_blit_fn)(void *user, int y, int h, const uint16_t *pixels);

// Scratch for one band: pixels + coverage + body position. Put it in internal
// SRAM — that is the entire point of banding.
size_t wr_scratch_bytes(int band_rows);

// One alpha byte per pixel. Built once at startup; PSRAM is fine, it is only
// ever read sequentially.
size_t wr_globe_bytes(void);
void wr_build_globe(uint8_t *mask);

void wr_init(wr_ctx *c, uint8_t *scratch, int band_rows, const uint8_t *globe);
void wr_draw_banded(wr_ctx *c, const wm_world *w, wr_blit_fn blit, void *user);

#endif  // WORMRENDER_H
