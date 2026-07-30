// wormrender — rasteriser for the worm and its word field.
//
// Renders in horizontal bands, not whole frames. The first version drew into a
// 466x466 PSRAM framebuffer: 286-475 ms a frame, against 28 ms to actually push
// the pixels out. The rasteriser makes several passes per frame and PSRAM is
// punishing for the scattered single-byte writes the coverage pass does. A
// band's pixels fit comfortably in internal SRAM, which is far quicker for that
// access pattern, and each band goes straight out over QSPI — no full
// framebuffer exists at all.
//
// Everything is float, deliberately: the S3's FPU is single-precision only, so
// double operations are software routines. That is the opposite of the call
// made in wormsim, whose determinism depends on float64 — but nothing in here
// feeds the simulation.

#include "wormrender.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// viewer/focus/worm-render.js uses WORM_BASE_RADIUS = 22 world units, but its
// camera frames the whole 800-unit body. Here the window is 400 units, so 22
// reads as a snake. 14 keeps the nematode slenderness at this magnification.
#define WR_DEFAULT_RADIUS 14.0f

size_t wr_scratch_bytes(int band_rows) {
    // band pixels (RGB565) + coverage byte + body-position byte
    return (size_t)WR_W * band_rows * (2 + 1 + 1);
}

static inline uint16_t to565(uint32_t c) {
    return (uint16_t)((((c >> 16) & 0xFF) >> 3) << 11 |
                      (((c >> 8) & 0xFF) >> 2) << 5 |
                      ((c & 0xFF) >> 3));
}

static inline void blend(uint16_t *px, uint32_t c, uint32_t a) {
    if (a == 0) return;
    if (a >= 255) { *px = to565(c); return; }
    uint32_t d = *px;
    uint32_t dr = (d >> 11) & 0x1F, dg = (d >> 5) & 0x3F, db = d & 0x1F;
    uint32_t sr = ((c >> 16) & 0xFF) >> 3, sg = ((c >> 8) & 0xFF) >> 2, sb = (c & 0xFF) >> 3;
    uint32_t ia = 255 - a;
    // (v * 257) >> 16 is v/255 to within a bit, without three integer divides.
    *px = (uint16_t)(((((sr * a + dr * ia) * 257) >> 16) << 11) |
                     ((((sg * a + dg * ia) * 257) >> 16) << 5) |
                     (((sb * a + db * ia) * 257) >> 16));
}

void wr_init(wr_ctx *c, uint8_t *scratch, int band_rows) {
    memset(c, 0, sizeof(*c));
    c->band_rows = band_rows;
    c->band = (uint16_t *)scratch;
    c->cov = scratch + (size_t)WR_W * band_rows * 2;
    c->seg = c->cov + (size_t)WR_W * band_rows;

    // 400 world units across = 2 x FOOD_SENSE_RADIUS, so the rim of the round
    // display sits exactly on the edge of what the worm can smell.
    c->view_units = 400.0f;
    c->body_radius = WR_DEFAULT_RADIUS;
    c->round_mask = true;
}

static inline float scale_of(const wr_ctx *c) { return (float)WR_W / c->view_units; }

// World y grows downward on screen: the scroller spawns words at y = 980 and
// decreases y to move them up the display (text_scroller.py SCROLL_SPEED).
static inline void w2s(const wr_ctx *c, float wx, float wy, float *sx, float *sy) {
    float s = scale_of(c);
    *sx = (wx - c->cam_x) * s + WR_W / 2.0f;
    *sy = (wy - c->cam_y) * s + WR_H / 2.0f;
}

// --- text -------------------------------------------------------------------

static int glyph_slot(const char *p, int len, int *consumed) {
    unsigned char ch = (unsigned char)p[0];
    if (ch == 0xE2 && len >= 3 && (unsigned char)p[1] == 0x80 &&
        (unsigned char)p[2] == 0x94) {
        *consumed = 3;  // U+2014 em dash
        return WR_GLYPH_EM_DASH;
    }
    *consumed = 1;
    if (ch < 32 || ch > 126) return -1;
    return ch - 32;
}

static float text_width(const char *s, int len) {
    float w = 0;
    for (int i = 0; i < len;) {
        int used, g = glyph_slot(s + i, len - i, &used);
        if (g >= 0) w += wr_font_glyph[g].adv;
        i += used;
    }
    return w;
}

// Centred on (cx, cy), matching the site's textAlign/textBaseline centre in
// viewer/focus/text-canvas.js. Rows outside the band are skipped.
static void draw_text(wr_ctx *c, int y0, int h, const char *s, int len,
                      float cx, float cy, uint32_t color, uint32_t alpha) {
    float pen = cx - text_width(s, len) / 2.0f;
    float base = cy + wr_font_size * 0.36f;  // optical centre, not the baseline

    for (int i = 0; i < len;) {
        int used, gi = glyph_slot(s + i, len - i, &used);
        i += used;
        if (gi < 0) continue;
        const wr_glyph *g = &wr_font_glyph[gi];
        if (g->w) {
            int gx0 = (int)lroundf(pen + g->bx);
            int gy0 = (int)lroundf(base + g->by);
            for (int gy = 0; gy < g->h; gy++) {
                int py = gy0 + gy - y0;
                if (py < 0 || py >= h) continue;
                const uint8_t *row =
                    wr_font_atlas + (size_t)(g->ay + gy) * wr_font_atlas_w + g->ax;
                uint16_t *dst = c->band + (size_t)py * WR_W;
                for (int gx = 0; gx < g->w; gx++) {
                    int px = gx0 + gx;
                    if (px < 0 || px >= WR_W) continue;
                    blend(dst + px, color, (uint32_t)row[gx] * alpha / 255);
                }
            }
        }
        pen += g->adv;
    }
}

static void band_words(wr_ctx *c, const wm_world *w, int y0, int h) {
    const wm_asset *a = w->a;
    const wm_scroller *sc = &w->scroller;
    const float margin = 40.0f;

    for (int li = 0; li < sc->n_lines; li++) {
        const wm_line *L = &sc->lines[li];
        for (uint16_t wi = 0; wi < L->n_words; wi++) {
            const wm_word *word = &L->w[wi];
            float sx, sy;
            w2s(c, (float)word->x, (float)word->y, &sx, &sy);
            if (sx < -margin || sx > WR_W + margin) continue;
            // A glyph reaches at most a font box either side of the centre.
            if (sy < y0 - wr_font_size || sy > y0 + h + wr_font_size) continue;

            uint32_t len;
            const char *txt = wm_str(&a->tok_text, word->tok, &len);

            // Same colour logic as the site: edible words read bright, inedible
            // set-dressing (speaker names, stage cues) recedes to blue-grey.
            uint32_t color, alpha;
            if (word->eaten) {
                color = WR_DIM;
                alpha = 60;  // ghost of a word already swallowed
            } else if (L->edible) {
                color = 0xFFFFFF;
                alpha = 178;
            } else {
                color = 0x96AACD;
                alpha = 115;
            }
            draw_text(c, y0, h, txt, (int)len, sx, sy, color, alpha);
        }
    }
}

// --- body -------------------------------------------------------------------

// viewer/focus/worm-render.js bodyProfile(), ported verbatim so the object and
// the website agree on what this animal looks like.
static float body_profile(float s) {
    const float head_len = 0.06f, tail_start = 0.72f;
    float r;
    if (s < head_len) {
        r = powf(s / head_len, 0.4f);  // rounded tip, not knife-pointed
    } else if (s > tail_start) {
        r = powf((1.0f - s) / (1.0f - tail_start), 1.2f);  // concave taper
    } else {
        r = 1.0f;
    }
    return r * (1.0f + 0.07f * sinf(s * (float)M_PI));  // gravid mid-body bulge
}

// Coverage for one segment, clipped to the band. Keeps the max coverage per
// pixel rather than accumulating: segments overlap heavily at the joints, and
// summing would leave a bright seam every 4 world units.
static void cover_segment(wr_ctx *c, int y0, int h, float ax, float ay,
                          float bx, float by, float rad, uint8_t seg) {
    if (rad < 0.35f) return;
    int minx = (int)floorf(fminf(ax, bx) - rad - 1), maxx = (int)ceilf(fmaxf(ax, bx) + rad + 1);
    int miny = (int)floorf(fminf(ay, by) - rad - 1), maxy = (int)ceilf(fmaxf(ay, by) + rad + 1);
    if (maxx < 0 || minx >= WR_W) return;
    miny -= y0;
    maxy -= y0;
    if (maxy < 0 || miny >= h) return;
    if (minx < 0) minx = 0;
    if (miny < 0) miny = 0;
    if (maxx >= WR_W) maxx = WR_W - 1;
    if (maxy >= h) maxy = h - 1;

    float dx = bx - ax, dy = by - ay;
    float len2 = dx * dx + dy * dy;
    // One reciprocal per segment rather than a divide per pixel; the inner loop
    // runs a few hundred thousand times a frame and division is not cheap here.
    float inv_len2 = len2 > 0.0f ? 1.0f / len2 : 0.0f;
    // Compare squared distances, and only take a square root in the one-pixel
    // band at the edge. Interior pixels dominate and each used to pay for a
    // sqrtf whose result was always going to clamp to full coverage anyway.
    float r_in = rad - 0.5f, r_out = rad + 0.5f;
    float r_in2 = r_in > 0.0f ? r_in * r_in : 0.0f, r_out2 = r_out * r_out;

    for (int py = miny; py <= maxy; py++) {
        uint8_t *crow = c->cov + (size_t)py * WR_W;
        uint8_t *srow = c->seg + (size_t)py * WR_W;
        float fy = (float)(py + y0) + 0.5f - ay;
        for (int px = minx; px <= maxx; px++) {
            float vx = (float)px + 0.5f - ax;
            float t = (vx * dx + fy * dy) * inv_len2;
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            float ex = vx - t * dx, ey = fy - t * dy;
            float d2 = ex * ex + ey * ey;
            if (d2 >= r_out2) continue;
            uint8_t v = (d2 <= r_in2) ? 255 : (uint8_t)((r_out - sqrtf(d2)) * 255.0f);
            if (v > crow[px]) {
                crow[px] = v;
                srow[px] = seg;
            }
        }
    }
}

static void band_worm(wr_ctx *c, int y0, int h, const float *px, const float *py,
                      const float *rad) {
    memset(c->cov, 0, (size_t)WR_W * h);
    memset(c->seg, 0, (size_t)WR_W * h);

    const int n = WM_N_SEGMENTS + 1;
    for (int i = 0; i < n - 1; i++)
        cover_segment(c, y0, h, px[i], py[i], px[i + 1], py[i + 1], rad[i],
                      (uint8_t)((float)i / (float)(n - 1) * 255.0f));

    // Composite with a head-to-tail gradient. The anterior is paler and cooler
    // (the pharynx catching light), deepening to the body green behind it,
    // which is what makes the eating end obvious on a screen this small without
    // gluing a separate highlight onto the nose.
    const uint32_t head_c = 0x9CF7C4, tail_c = WR_ACCENT;
    for (int y = 0; y < h; y++) {
        uint8_t *crow = c->cov + (size_t)y * WR_W;
        uint8_t *srow = c->seg + (size_t)y * WR_W;
        uint16_t *drow = c->band + (size_t)y * WR_W;
        for (int x = 0; x < WR_W; x++) {
            if (!crow[x]) continue;
            uint32_t m = srow[x] * 255u / 46u;  // ramp over the front ~18%
            if (m > 255u) m = 255u;
            uint32_t col =
                ((((head_c >> 16 & 0xFF) * (255 - m) + (tail_c >> 16 & 0xFF) * m) / 255) << 16) |
                ((((head_c >> 8 & 0xFF) * (255 - m) + (tail_c >> 8 & 0xFF) * m) / 255) << 8) |
                (((head_c & 0xFF) * (255 - m) + (tail_c & 0xFF) * m) / 255);
            blend(drow + x, col, crow[x]);
        }
    }
}

// One sqrt per row rather than a hypot per pixel: the visible span of a circle
// on a given row is closed-form, and the corners just get zeroed.
static void band_mask(wr_ctx *c, int y0, int h) {
    const float cx = WR_W / 2.0f, cy = WR_H / 2.0f, r = WR_W / 2.0f - 0.5f;
    for (int y = 0; y < h; y++) {
        uint16_t *row = c->band + (size_t)y * WR_W;
        float dy = (float)(y + y0) + 0.5f - cy;
        float half = r * r - dy * dy;
        int x0 = 0, x1 = -1;
        if (half > 0.0f) {
            float hx = sqrtf(half);
            x0 = (int)(cx - hx);
            x1 = (int)(cx + hx);
            if (x0 < 0) x0 = 0;
            if (x1 >= WR_W) x1 = WR_W - 1;
        }
        for (int x = 0; x < x0; x++) row[x] = 0;
        for (int x = x1 + 1; x < WR_W; x++) row[x] = 0;
    }
}

void wr_draw_banded(wr_ctx *c, const wm_world *w, wr_blit_fn blit, void *user) {
    c->cam_x = (float)w->body.target_x;
    c->cam_y = (float)w->body.target_y;

    // Project the midline once for the whole frame. It is 201 points, and
    // reprojecting it per band would repeat that work for every one of them.
    // midline() is the head point followed by every segment tail.
    static float px[WM_N_SEGMENTS + 1], py[WM_N_SEGMENTS + 1];
    const wm_body *b = &w->body;
    w2s(c, (float)b->hx[0], (float)b->hy[0], &px[0], &py[0]);
    for (int i = 0; i < WM_N_SEGMENTS; i++)
        w2s(c, (float)b->tx[i], (float)b->ty[i], &px[i + 1], &py[i + 1]);

    // Per-segment radius, computed once for the whole frame.
    static float rad[WM_N_SEGMENTS];
    float sc = scale_of(c);
    for (int i = 0; i < WM_N_SEGMENTS; i++)
        rad[i] = c->body_radius * body_profile((float)i / (float)WM_N_SEGMENTS) * sc;

    const uint16_t stage = to565(WR_STAGE);

    for (int y0 = 0; y0 < WR_H; y0 += c->band_rows) {
        int h = WR_H - y0 < c->band_rows ? WR_H - y0 : c->band_rows;

        size_t n = (size_t)WR_W * h;
        for (size_t i = 0; i < n; i++) c->band[i] = stage;
        band_words(c, w, y0, h);
        band_worm(c, y0, h, px, py, rad);
        if (c->round_mask) band_mask(c, y0, h);

        blit(user, y0, h, c->band);
    }
}
