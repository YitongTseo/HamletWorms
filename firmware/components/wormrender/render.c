// wormrender — rasteriser for the worm and its word field.

#include "wormrender.h"

#include <math.h>

// float, deliberately — the opposite call from wormsim.
//
// The S3 has a single-precision FPU and no float unit, so every double
// operation is a software routine. The first version of this file ran the
// rasteriser in float and took 2.7 s a frame: ~400k emulated sqrt/hypot calls
// between the segment coverage pass and the round mask. Nothing here feeds the
// simulation, so nothing here needs float64.
#include <stdlib.h>
#include <string.h>

// viewer/focus/worm-render.js uses WORM_BASE_RADIUS = 22 world units, but its
// camera frames the whole 800-unit body. Here the window is 400 units, so 22
// fills a third of the screen with worm. 14 keeps the nematode slenderness at
// this magnification. Overridable per-context.
#define WR_DEFAULT_RADIUS 14.0

// Full-screen 8-bit coverage buffer for the body. Segments overlap heavily at
// the joints, so alpha-accumulating them directly would leave a bright seam
// every 4 world units. Taking the max coverage per pixel and compositing once
// removes the seams and costs one byte per pixel.
//
// Plus a parallel buffer holding position along the body (0 = head, 255 = tail)
// for whichever segment won the coverage test, so the composite can shade
// head-to-tail in one pass instead of bolting a separate blob onto the nose.
//
// Caller-owned (wr_init): 434 KB of .bss overflows the S3's internal DRAM.
static uint8_t *wr_cov;
static uint8_t *wr_seg;

size_t wr_scratch_bytes(void) { return (size_t)WR_W * WR_H * 2; }

static inline uint16_t to565(uint32_t c) {
    return (uint16_t)((((c >> 16) & 0xFF) >> 3) << 11 |
                      (((c >> 8) & 0xFF) >> 2) << 5 |
                      ((c & 0xFF) >> 3));
}

// Blend `c` over the framebuffer at `a`/255. Done in 565 component space —
// close enough at this size and much cheaper than round-tripping to 888.
static inline void blend(uint16_t *px, uint32_t c, uint32_t a) {
    if (a == 0) return;
    if (a >= 255) { *px = to565(c); return; }
    uint32_t d = *px;
    uint32_t dr = (d >> 11) & 0x1F, dg = (d >> 5) & 0x3F, db = d & 0x1F;
    uint32_t sr = ((c >> 16) & 0xFF) >> 3, sg = ((c >> 8) & 0xFF) >> 2, sb = (c & 0xFF) >> 3;
    uint32_t ia = 255 - a;
    // (v * 257) >> 16 is v/255 to within a bit, and avoids three divides on a
    // core that has no hardware integer divide worth the name.
    *px = (uint16_t)(((((sr * a + dr * ia) * 257) >> 16) << 11) |
                     ((((sg * a + dg * ia) * 257) >> 16) << 5) |
                     (((sb * a + db * ia) * 257) >> 16));
}

void wr_init(wr_ctx *c, uint16_t *framebuffer, uint8_t *scratch) {
    memset(c, 0, sizeof(*c));
    c->fb = framebuffer;
    wr_cov = scratch;
    wr_seg = scratch + (size_t)WR_W * WR_H;
    // 400 world units across = 2 x FOOD_SENSE_RADIUS, so the rim of the round
    // display sits exactly on the edge of what the worm can smell.
    c->view_units = 400.0;
    c->body_radius = WR_DEFAULT_RADIUS;
    // At 400 units the ring lands exactly on the rim of the panel, so drawing
    // it is redundant — the bezel is the horizon. Only worth it when zoomed out.
    c->show_smell_ring = false;
    c->round_mask = true;
}

void wr_clear(wr_ctx *c, uint32_t rgb888) {
    uint16_t v = to565(rgb888);
    for (int i = 0; i < WR_W * WR_H; i++) c->fb[i] = v;
}

static inline float scale_of(const wr_ctx *c) { return (float)WR_W / c->view_units; }

// World y grows downward on screen: the scroller spawns words at y = 980 and
// decreases y to move them up the display (text_scroller.py SCROLL_SPEED).
static inline void w2s(const wr_ctx *c, float wx, float wy, float *sx, float *sy) {
    float s = scale_of(c);
    *sx = (wx - c->cam_x) * s + WR_W / 2.0;
    *sy = (wy - c->cam_y) * s + WR_H / 2.0;
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

// Draws `s` centred on (cx, cy), matching the site's textAlign/textBaseline
// centre in viewer/focus/text-canvas.js.
static void draw_text(wr_ctx *c, const char *s, int len, float cx, float cy,
                      uint32_t color, uint32_t alpha) {
    float pen = cx - text_width(s, len) / 2.0;
    float base = cy + wr_font_size * 0.36;  // optical centre, not the baseline

    for (int i = 0; i < len;) {
        int used, gi = glyph_slot(s + i, len - i, &used);
        i += used;
        if (gi < 0) continue;
        const wr_glyph *g = &wr_font_glyph[gi];
        if (g->w) {
            int x0 = (int)lroundf(pen + g->bx);
            int y0 = (int)lroundf(base + g->by);
            for (int gy = 0; gy < g->h; gy++) {
                int py = y0 + gy;
                if (py < 0 || py >= WR_H) continue;
                const uint8_t *row = wr_font_atlas + (size_t)(g->ay + gy) * wr_font_atlas_w + g->ax;
                uint16_t *dst = c->fb + (size_t)py * WR_W;
                for (int gx = 0; gx < g->w; gx++) {
                    int px = x0 + gx;
                    if (px < 0 || px >= WR_W) continue;
                    blend(dst + px, color, (uint32_t)row[gx] * alpha / 255);
                }
            }
        }
        pen += g->adv;
    }
}

void wr_draw_words(wr_ctx *c, const wm_world *w) {
    const wm_asset *a = w->a;
    const wm_scroller *sc = &w->scroller;
    float s = scale_of(c);
    float margin = 40.0;

    for (int li = 0; li < sc->n_lines; li++) {
        const wm_line *L = &sc->lines[li];
        for (uint16_t wi = 0; wi < L->n_words; wi++) {
            const wm_word *word = &L->w[wi];
            float sx, sy;
            w2s(c, word->x, word->y, &sx, &sy);
            if (sx < -margin || sx > WR_W + margin || sy < -margin || sy > WR_H + margin)
                continue;

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
            draw_text(c, txt, (int)len, sx, sy, color, alpha);
        }
    }
    (void)s;
}

// --- body -------------------------------------------------------------------

// viewer/focus/worm-render.js bodyProfile(), ported verbatim so the object and
// the website agree on what this animal looks like.
static float body_profile(float s) {
    const float head_len = 0.06, tail_start = 0.72;
    float r;
    if (s < head_len) {
        r = powf(s / head_len, 0.4);  // rounded tip, not knife-pointed
    } else if (s > tail_start) {
        r = powf((1.0 - s) / (1.0 - tail_start), 1.2);  // concave taper to a point
    } else {
        r = 1.0;
    }
    return r * (1.0 + 0.07 * sinf(s * (float)M_PI));  // gravid mid-body bulge
}

static void cover_segment(float x0, float y0, float x1, float y1, float rad,
                          uint8_t seg) {
    if (rad < 0.35) return;
    int minx = (int)floorf(fminf(x0, x1) - rad - 1), maxx = (int)ceilf(fmaxf(x0, x1) + rad + 1);
    int miny = (int)floorf(fminf(y0, y1) - rad - 1), maxy = (int)ceilf(fmaxf(y0, y1) + rad + 1);
    if (maxx < 0 || minx >= WR_W || maxy < 0 || miny >= WR_H) return;
    if (minx < 0) minx = 0;
    if (miny < 0) miny = 0;
    if (maxx >= WR_W) maxx = WR_W - 1;
    if (maxy >= WR_H) maxy = WR_H - 1;

    float dx = x1 - x0, dy = y1 - y0;
    float len2 = dx * dx + dy * dy;

    for (int py = miny; py <= maxy; py++) {
        uint8_t *crow = wr_cov + (size_t)py * WR_W;
        for (int px = minx; px <= maxx; px++) {
            float vx = px + 0.5 - x0, vy = py + 0.5 - y0;
            float t = len2 > 0.0 ? (vx * dx + vy * dy) / len2 : 0.0;
            t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
            float ex = vx - t * dx, ey = vy - t * dy;
            float d = sqrtf(ex * ex + ey * ey);
            float cov = rad + 0.5 - d;  // 1 px of analytic edge falloff
            if (cov <= 0.0) continue;
            if (cov > 1.0) cov = 1.0;
            uint8_t v = (uint8_t)(cov * 255.0);
            if (v > crow[px]) {  // max, not sum — no joint seams
                crow[px] = v;
                wr_seg[(size_t)py * WR_W + px] = seg;
            }
        }
    }
}

void wr_draw_worm(wr_ctx *c, const wm_world *w) {
    const wm_body *b = &w->body;
    float s = scale_of(c);

    // midline() is the head point followed by every segment tail: 201 points.
    static float px[WM_N_SEGMENTS + 1], py[WM_N_SEGMENTS + 1];
    w2s(c, b->hx[0], b->hy[0], &px[0], &py[0]);
    for (int i = 0; i < WM_N_SEGMENTS; i++)
        w2s(c, b->tx[i], b->ty[i], &px[i + 1], &py[i + 1]);

    memset(wr_cov, 0, (size_t)WR_W * WR_H);
    memset(wr_seg, 0, (size_t)WR_W * WR_H);

    int n = WM_N_SEGMENTS + 1;
    for (int i = 0; i < n - 1; i++) {
        float t = (float)i / (float)(n - 1);
        float rad = c->body_radius * body_profile(t) * s;
        cover_segment(px[i], py[i], px[i + 1], py[i + 1], rad, (uint8_t)(t * 255.0));
    }

    // Composite with a head-to-tail gradient. The anterior is paler and cooler
    // (the pharynx region catching light), deepening to the body green behind
    // it — which is what makes the eating end obvious on a screen this small
    // without gluing a separate highlight onto the nose.
    const uint32_t head_c = 0x9CF7C4, tail_c = WR_ACCENT;
    for (int y = 0; y < WR_H; y++) {
        uint8_t *crow = wr_cov + (size_t)y * WR_W;
        uint8_t *srow = wr_seg + (size_t)y * WR_W;
        uint16_t *drow = c->fb + (size_t)y * WR_W;
        for (int x = 0; x < WR_W; x++) {
            if (!crow[x]) continue;
            // Ramp over the front ~18% of the body, flat green after that.
            uint32_t m = srow[x] * 255u / 46u;
            if (m > 255u) m = 255u;
            uint32_t col =
                ((((head_c >> 16 & 0xFF) * (255 - m) + (tail_c >> 16 & 0xFF) * m) / 255) << 16) |
                ((((head_c >> 8 & 0xFF) * (255 - m) + (tail_c >> 8 & 0xFF) * m) / 255) << 8) |
                (((head_c & 0xFF) * (255 - m) + (tail_c & 0xFF) * m) / 255);
            blend(drow + x, col, crow[x]);
        }
    }
}

// --- frame ------------------------------------------------------------------

static void draw_ring(wr_ctx *c, float world_radius, uint32_t color, uint32_t alpha) {
    float r = world_radius * scale_of(c);
    float cx = WR_W / 2.0, cy = WR_H / 2.0;
    // The camera tracks the head, so a ring at FOOD_SENSE_RADIUS is centred.
    for (int y = 0; y < WR_H; y++) {
        for (int x = 0; x < WR_W; x++) {
            float d = fabsf(hypotf(x + 0.5 - cx, y + 0.5 - cy) - r);
            if (d > 1.0) continue;
            blend(c->fb + (size_t)y * WR_W + x, color, (uint32_t)((1.0 - d) * alpha));
        }
    }
}

static void apply_round_mask(wr_ctx *c) {
    const float cx = WR_W / 2.0f, cy = WR_H / 2.0f, r = WR_W / 2.0f - 0.5f;
    for (int y = 0; y < WR_H; y++) {
        uint16_t *row = c->fb + (size_t)y * WR_W;
        float dy = y + 0.5f - cy;
        float half = r * r - dy * dy;
        // One sqrt per row rather than a hypot per pixel: the span of a circle
        // on a given row is closed-form, and the corners just get memset.
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

void wr_draw(wr_ctx *c, const wm_world *w) {
    c->cam_x = w->body.target_x;
    c->cam_y = w->body.target_y;

    wr_clear(c, WR_STAGE);
    if (c->show_smell_ring) draw_ring(c, WM_FOOD_SENSE_RADIUS, WR_ACCENT, 40);
    wr_draw_words(c, w);
    wr_draw_worm(c, w);
    if (c->round_mask) apply_round_mask(c);
}
