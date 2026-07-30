// wormrender — draws one worm and its word field into an RGB565 framebuffer.
//
// Portable C, no ESP-IDF: tools/preview builds it on the host and writes PNGs,
// so the look can be settled before the hardware is wired up. The board just
// hands it the PSRAM framebuffer that goes out over QSPI.
//
// Framing: the visible circle is the worm's chemosensory horizon. World.
// FOOD_SENSE_RADIUS is 200 units, so a 400-unit window mapped onto the 466 px
// round display means the edge of the screen is exactly the edge of what the
// worm can smell. Words drift in from outside its awareness and become real as
// they cross the rim.
//
// The body is 200 segments x 4.0 units = 800 units long, twice the window, so
// the tail trails out of frame. That is deliberate — the head is where the
// eating happens, and a microscope does not show you the whole animal.

#ifndef WORMRENDER_H
#define WORMRENDER_H

#include <stdbool.h>
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
    uint16_t *fb;  // WR_W * WR_H, RGB565 big-endian for the CO5300

    // World units across the display. 400 puts the rim on the smell radius.
    double view_units;

    // Camera centre, world units. Normally the worm's head.
    double cam_x, cam_y;

    // Body half-width in world units. The site uses 22 for a camera that sees
    // the whole animal; at this magnification that reads as a snake rather than
    // a nematode, so the object runs slimmer.
    double body_radius;

    bool show_smell_ring;  // only visible if view_units > 2*FOOD_SENSE_RADIUS
    bool round_mask;       // black the corners outside the circular panel
} wr_ctx;

void wr_init(wr_ctx *c, uint16_t *framebuffer);
void wr_draw(wr_ctx *c, const wm_world *w);

// Exposed for the preview harness.
void wr_clear(wr_ctx *c, uint32_t rgb888);
void wr_draw_words(wr_ctx *c, const wm_world *w);
void wr_draw_worm(wr_ctx *c, const wm_world *w);

#endif  // WORMRENDER_H
