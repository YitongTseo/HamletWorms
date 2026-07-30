// wormrender — draws one worm and its word field, band by band.
//
// Portable C, no ESP-IDF: tools/preview builds it on the host and writes PNGs,
// so the look can be settled before the hardware is wired up.
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
// Output goes out a band at a time through a caller-supplied blit, rather than
// into a framebuffer. On the board that means the per-pixel work happens in
// internal SRAM and each band DMAs straight to the panel; a 466x466 PSRAM
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

// Palette, from viewer/palette.js "poetry", so the object matches the site.
#define WR_STAGE 0x03140a   // near-black green — the canvas background
#define WR_ACCENT 0x3ddc84  // the worm
#define WR_FG 0xc6f6d5
#define WR_DIM 0x5a8f6a
#define WR_WARM 0xffcc66
#define WR_HOT 0xff6b6b

typedef struct {
    uint16_t *band;  // WR_W x band_rows, RGB565
    uint8_t *cov;    // per-pixel body coverage
    uint8_t *seg;    // per-pixel position along the body, for the gradient
    int band_rows;

    // World units across the display. 400 puts the rim on the smell radius.
    float view_units;

    // Camera centre, world units. Normally the worm's head.
    float cam_x, cam_y;

    // Body half-width in world units. The site uses 22 for a camera that sees
    // the whole animal; at this magnification that reads as a snake.
    float body_radius;

    bool round_mask;  // zero the corners outside the circular panel
} wr_ctx;

// Called once per band with rows [y, y + h) of the frame.
typedef void (*wr_blit_fn)(void *user, int y, int h, const uint16_t *pixels);

// Scratch for one band: pixels + coverage + body position. Put it in internal
// SRAM — that is the entire point of banding.
size_t wr_scratch_bytes(int band_rows);
void wr_init(wr_ctx *c, uint8_t *scratch, int band_rows);
void wr_draw_banded(wr_ctx *c, const wm_world *w, wr_blit_fn blit, void *user);

#endif  // WORMRENDER_H
