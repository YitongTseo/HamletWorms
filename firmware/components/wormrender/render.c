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
//
// Draw order, back to front: globe graticule, words, worm, flash.

#include "wormrender.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Per-stage microseconds, for tuning. Costs nothing while wr_clock is NULL.
uint32_t wr_us_frame, wr_us_words, wr_us_worm;
uint32_t (*wr_clock)(void);
#define WR_T() (wr_clock ? wr_clock() : 0u)

// viewer/focus/worm-render.js uses WORM_BASE_RADIUS = 22 world units, but its
// camera frames the whole 800-unit body. Here the window is 400 units, so 22
// reads as a snake. 15 keeps the nematode slenderness at this magnification.
#define WR_DEFAULT_RADIUS 15.0f

size_t wr_scratch_bytes(int band_rows) {
    // two band buffers (RGB565) + coverage byte + body-position byte
    return (size_t)WR_W * band_rows * (2 + 2 + 1 + 1);
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

// Body colour by position along the animal, rebuilt only when the flash level
// changes.
static uint16_t body_lut[256];
static uint32_t body_lut_key = 0xFFFFFFFFu;

static inline uint32_t lerp_rgb(uint32_t a, uint32_t b, uint32_t m) {
    uint32_t ia = 255 - m;
    return ((((a >> 16 & 0xFF) * ia + (b >> 16 & 0xFF) * m) / 255) << 16) |
           ((((a >> 8 & 0xFF) * ia + (b >> 8 & 0xFF) * m) / 255) << 8) |
           (((a & 0xFF) * ia + (b & 0xFF) * m) / 255);
}

void wr_init(wr_ctx *c, uint8_t *scratch, int band_rows, const uint8_t *globe) {
    memset(c, 0, sizeof(*c));
    c->band_rows = band_rows;
    c->band_mem = (uint16_t *)scratch;
    c->band = c->band_mem;
    c->band_parity = 0;
    c->cov = scratch + (size_t)WR_W * band_rows * 4;
    c->seg = c->cov + (size_t)WR_W * band_rows;
    c->globe = globe;

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

// --- the frame ---------------------------------------------------------------
//
// A wireframe ellipsoid inscribed in the simulation's own rectangle, drawn in
// WORLD space rather than screen space.
//
// The previous version was pinned to the display, which looked like a globe but
// told you nothing: it slid along with the camera, so the worm was forever at
// its centre. Anchoring it to the world turns it into an instrument. Its poles
// sit on the top and bottom of the box the words live in — the line at
// SPAWN_Y where they appear and the line at KILL_Y where they vanish — so the
// curvature tells you where in that box the worm currently is, and the words
// visibly rise past it.
//
// Extents come from the simulation constants, not from taste: x spans the full
// WM_WORLD_W, y spans KILL_Y..SPAWN_Y.

#define WR_FRAME_CX (WM_WORLD_W * 0.5f)
#define WR_FRAME_CY ((float)((WM_SPAWN_Y + WM_KILL_Y) * 0.5))
#define WR_FRAME_A (WM_WORLD_W * 0.5f)
#define WR_FRAME_B ((float)((WM_SPAWN_Y - WM_KILL_Y) * 0.5))
#define WR_FRAME_TILT 0.30f

#define WR_FRAME_PARALLELS 7
#define WR_FRAME_MERIDIANS 8
#define WR_FRAME_STEPS 64
#define WR_FRAME_MAX_PTS ((WR_FRAME_PARALLELS + WR_FRAME_MERIDIANS) * (WR_FRAME_STEPS + 1))

typedef struct {
    float wx, wy;   // world position
    uint8_t alpha;  // baked from depth: the far side sits behind the near side
    uint8_t start;  // 1 = begins a new polyline
} wr_frame_pt;

static wr_frame_pt wr_frame[WR_FRAME_MAX_PTS];
static int wr_frame_n;

size_t wr_globe_bytes(void) { return 0; }  // no mask any more

void wr_build_globe(uint8_t *unused) {
    (void)unused;
    const float ct = cosf(WR_FRAME_TILT), st = sinf(WR_FRAME_TILT);
    // Undo the foreshortening the tilt introduces, so the poles still land
    // exactly on SPAWN_Y and KILL_Y.
    const float b = WR_FRAME_B / ct;
    const float cdepth = WR_FRAME_B;
    wr_frame_n = 0;

    for (int p = 0; p < WR_FRAME_PARALLELS; p++) {
        float lat = (-1.0f + 2.0f * (float)(p + 1) / (float)(WR_FRAME_PARALLELS + 1))
                    * (float)M_PI * 0.5f;
        float clat = cosf(lat), slat = sinf(lat);
        for (int i = 0; i <= WR_FRAME_STEPS; i++) {
            float lon = (float)i * (2.0f * (float)M_PI / (float)WR_FRAME_STEPS);
            float x = WR_FRAME_A * clat * cosf(lon);
            float y = b * slat;
            float z = cdepth * clat * sinf(lon);
            float yr = y * ct - z * st, zr = y * st + z * ct;
            float d = zr / cdepth;
            if (d < -1.0f) d = -1.0f;
            if (d > 1.0f) d = 1.0f;
            wr_frame[wr_frame_n++] = (wr_frame_pt){
                WR_FRAME_CX + x, WR_FRAME_CY + yr,
                (uint8_t)(64.0f + 96.0f * (d * 0.5f + 0.5f)), i == 0};
        }
    }
    for (int m = 0; m < WR_FRAME_MERIDIANS; m++) {
        float lon = (float)m * ((float)M_PI / (float)WR_FRAME_MERIDIANS);
        float clon = cosf(lon), slon = sinf(lon);
        for (int i = 0; i <= WR_FRAME_STEPS; i++) {
            float lat = -(float)M_PI * 0.5f + (float)i * ((float)M_PI / (float)WR_FRAME_STEPS);
            float clat = cosf(lat);
            float x = WR_FRAME_A * clat * clon;
            float y = b * sinf(lat);
            float z = cdepth * clat * slon;
            float yr = y * ct - z * st, zr = y * st + z * ct;
            float d = zr / cdepth;
            if (d < -1.0f) d = -1.0f;
            if (d > 1.0f) d = 1.0f;
            wr_frame[wr_frame_n++] = (wr_frame_pt){
                WR_FRAME_CX + x, WR_FRAME_CY + yr,
                (uint8_t)(64.0f + 96.0f * (d * 0.5f + 0.5f)), i == 0};
        }
    }
}

// Thick anti-aliased line, walked along its major axis.
//
// The obvious implementation — bounding box, distance to segment per pixel — is
// what the body capsules use, and it is fine for them because they are 5 px
// long. The frame's segments are ~90 px and mostly diagonal, where the box is
// 90x90 to cover a line 3 px wide: about 8000 pixel tests for 270 pixels of
// line. Walking the major axis instead touches only the pixels near the line.
//
// Crossings blend twice, which brightens the intersections slightly. On a
// graticule that reads as nodes, so it is left alone rather than routed through
// a coverage buffer.
static void frame_line(wr_ctx *c, int y0, int h, float ax, float ay,
                       float bx, float by, float half, uint32_t alpha) {
    float dx = bx - ax, dy = by - ay;
    float len = sqrtf(dx * dx + dy * dy);
    if (len < 0.01f) return;

    if (fabsf(dx) >= fabsf(dy)) {
        if (ax > bx) { float t; t = ax; ax = bx; bx = t; t = ay; ay = by; by = t; dx = -dx; dy = -dy; }
        // Half-open in the major axis: pixel centres in [ax, bx). Consecutive
        // segments of a polyline then tile exactly instead of both painting the
        // shared endpoint, which blended twice and left a bead at every joint.
        int x0 = (int)ceilf(ax - 0.5f), x1 = (int)ceilf(bx - 0.5f) - 1;
        if (x1 < 0 || x0 >= WR_W || x1 < x0) return;
        if (x0 < 0) x0 = 0;
        if (x1 >= WR_W) x1 = WR_W - 1;
        // Perpendicular distance from a vertical offset, for this slope.
        float k = fabsf(dx) / len;
        float span = (half + 0.5f) / (k > 0.01f ? k : 0.01f);
        float slope = dy / dx;
        for (int px = x0; px <= x1; px++) {
            float yc = ay + slope * ((float)px + 0.5f - ax);
            int ya = (int)floorf(yc - span), yb = (int)ceilf(yc + span);
            if (ya < y0) ya = y0;
            if (yb >= y0 + h) yb = y0 + h - 1;
            for (int py = ya; py <= yb; py++) {
                float cov = half + 0.5f - fabsf((float)py + 0.5f - yc) * k;
                if (cov <= 0.0f) continue;
                if (cov > 1.0f) cov = 1.0f;
                blend(c->band + (size_t)(py - y0) * WR_W + px, WR_GLOBE,
                      (uint32_t)(cov * alpha));
            }
        }
    } else {
        if (ay > by) { float t; t = ax; ax = bx; bx = t; t = ay; ay = by; by = t; dx = -dx; dy = -dy; }
        int ya = (int)ceilf(ay - 0.5f), yb = (int)ceilf(by - 0.5f) - 1;
        if (yb < y0 || ya >= y0 + h || yb < ya) return;
        if (ya < y0) ya = y0;
        if (yb >= y0 + h) yb = y0 + h - 1;
        float k = fabsf(dy) / len;
        float span = (half + 0.5f) / (k > 0.01f ? k : 0.01f);
        float slope = dx / dy;
        for (int py = ya; py <= yb; py++) {
            float xc = ax + slope * ((float)py + 0.5f - ay);
            int x0 = (int)floorf(xc - span), x1 = (int)ceilf(xc + span);
            if (x0 < 0) x0 = 0;
            if (x1 >= WR_W) x1 = WR_W - 1;
            uint16_t *drow = c->band + (size_t)(py - y0) * WR_W;
            for (int px = x0; px <= x1; px++) {
                float cov = half + 0.5f - fabsf((float)px + 0.5f - xc) * k;
                if (cov <= 0.0f) continue;
                if (cov > 1.0f) cov = 1.0f;
                blend(drow + px, WR_GLOBE, (uint32_t)(cov * alpha));
            }
        }
    }
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

static float text_width_t(const char *s, int len, float tracking) {
    float w = 0;
    for (int i = 0; i < len;) {
        int used, g = glyph_slot(s + i, len - i, &used);
        if (g >= 0) w += wr_font_glyph[g].adv + tracking;
        i += used;
    }
    return w > 0 ? w - tracking : 0;
}


// Centred on (cx, cy), matching the site's textAlign/textBaseline centre in
// viewer/focus/text-canvas.js. Rows outside the band are skipped.
static void draw_text_t(wr_ctx *c, int y0, int h, const char *s, int len,
                        float cx, float cy, uint32_t color, uint32_t alpha,
                        float tracking) {
    float pen = cx - text_width_t(s, len, tracking) / 2.0f;
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
                    blend(dst + px, color, ((uint32_t)row[gx] * alpha * 257) >> 16);
                }
            }
        }
        pen += g->adv + tracking;
    }
}

static void draw_text(wr_ctx *c, int y0, int h, const char *s, int len,
                      float cx, float cy, uint32_t color, uint32_t alpha) {
    draw_text_t(c, y0, h, s, len, cx, cy, color, alpha, 0.0f);
}

static void band_words(wr_ctx *c, const wm_world *w, int y0, int h) {
    const wm_asset *a = w->a;
    const wm_scroller *sc = &w->scroller;
    const float margin = 60.0f;

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

            // Edible words read bright; inedible set-dressing (speaker names,
            // stage cues) recedes, the way it does on the site.
            uint32_t color, alpha;
            if (word->eaten) {
                color = WR_GHOST;
                alpha = 90;  // still there, but spent
            } else if (L->edible) {
                color = WR_WORD;
                alpha = 255;
            } else {
                color = WR_WORD_INERT;
                alpha = 120;
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
            // Segments sit ~5 px apart with a ~17 px radius, so every interior
            // pixel is covered by about seven capsules and saturated by the
            // first one to reach it. Nothing later can raise it, so skip the
            // arithmetic entirely — that is most of the animal.
            if (crow[px] == 255) continue;
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

// A filled disc, used for the neurons firing along the body.
static void dot(wr_ctx *c, int y0, int h, float x, float y, float r,
                uint32_t color, uint32_t alpha) {
    int miny = (int)floorf(y - r - 1) - y0, maxy = (int)ceilf(y + r + 1) - y0;
    int minx = (int)floorf(x - r - 1), maxx = (int)ceilf(x + r + 1);
    if (maxy < 0 || miny >= h || maxx < 0 || minx >= WR_W) return;
    if (miny < 0) miny = 0;
    if (maxy >= h) maxy = h - 1;
    if (minx < 0) minx = 0;
    if (maxx >= WR_W) maxx = WR_W - 1;
    for (int py = miny; py <= maxy; py++) {
        uint16_t *drow = c->band + (size_t)py * WR_W;
        float dy = (float)(py + y0) + 0.5f - y;
        for (int px = minx; px <= maxx; px++) {
            float dx = (float)px + 0.5f - x;
            float d = sqrtf(dx * dx + dy * dy);
            float cov = r + 0.5f - d;
            if (cov <= 0.0f) continue;
            if (cov > 1.0f) cov = 1.0f;
            blend(drow + px, color, (uint32_t)(cov * alpha));
        }
    }
}

// The pulse that leaves the head when a word goes in. Drawn from the row-wise
// closed form, so it costs a handful of pixels per row rather than a pass.
static void pulse_ring(wr_ctx *c, int y0, int h, float r, uint32_t alpha) {
    const float cx = WR_W * 0.5f, cy = WR_H * 0.5f;
    for (int py = 0; py < h; py++) {
        float dy = (float)(py + y0) + 0.5f - cy;
        float q = r * r - dy * dy;
        if (q <= 0.0f) continue;
        float dx = sqrtf(q);
        uint16_t *drow = c->band + (size_t)py * WR_W;
        for (int side = 0; side < 2; side++) {
            float xc = side ? cx + dx : cx - dx;
            int x0 = (int)(xc - 2.0f), x1 = (int)(xc + 2.0f);
            for (int px = x0; px <= x1; px++) {
                if (px < 0 || px >= WR_W) continue;
                float d = fabsf((float)px + 0.5f - xc);
                if (d > 1.5f) continue;
                blend(drow + px, WR_FIRE, (uint32_t)(alpha * (1.0f - d / 1.5f)));
            }
        }
    }
}

// Overlay the connectome on the moving animal. Each neuron has a precomputed
// (axial, lateral) anchor; sample the midline at its axial position, take the
// local perpendicular, and place it there — so when the worm twists, the
// neurons twist with it. Brightness is the neuron's actual charge, so what
// lights up is what the simulation is really doing.
static void band_xray(wr_ctx *c, const wm_world *w, int y0, int h,
                      const float *px, const float *py, const float *rad) {
    const wm_asset *a = w->a;
    if (!a->neuron_axial) return;
    const int n = WM_N_SEGMENTS + 1;
    const wm_brain *b = &w->brain;

    for (uint32_t i = 0; i < a->n_neurons; i++) {
        float ax = a->neuron_axial[i];
        if (ax < 0.0f) continue;

        int k = (int)(ax * (float)(n - 2));
        if (k < 0) k = 0;
        if (k > n - 2) k = n - 2;

        float tx = px[k + 1] - px[k], ty = py[k + 1] - py[k];
        float tl = sqrtf(tx * tx + ty * ty);
        if (tl < 1e-4f) { tx = 1.0f; ty = 0.0f; tl = 1.0f; }
        // perpendicular to the local tangent, scaled by how wide the body is here
        float off = a->neuron_lateral[i] * rad[k] * 0.78f;
        float nx = px[k] - ty / tl * off;
        float ny = py[k] + tx / tl * off;

        // psyn is double-buffered; this_state holds the charge the brain will
        // next fire on. FIRE_THRESHOLD is the interesting scale.
        double v = b->psyn[(size_t)i * 2 + b->this_state];
        float lit = (float)(v / WM_FIRE_THRESHOLD);
        if (lit < 0.0f) lit = 0.0f;
        if (lit > 1.0f) lit = 1.0f;

        uint32_t col = lerp_rgb(0x2F6B57, WR_FIRE, (uint32_t)(lit * 255.0f));
        uint32_t alpha = (uint32_t)(90.0f + 165.0f * lit);
        dot(c, y0, h, nx, ny, 1.3f + 1.5f * lit, col, alpha);
    }
}

static void band_worm(wr_ctx *c, int y0, int h, const float *px, const float *py,
                      const float *rad) {
    memset(c->cov, 0, (size_t)WR_W * h);
    memset(c->seg, 0, (size_t)WR_W * h);

    // One capsule per segment. Striding them in pairs saved about a third of
    // the raster but left visible facets on the outer edge of the head curl,
    // where the midline turns faster than the radius can hide.
    const int n = WM_N_SEGMENTS + 1;
    for (int i = 0; i < n - 1; i++)
        cover_segment(c, y0, h, px[i], py[i], px[i + 1], py[i + 1], rad[i],
                      (uint8_t)((float)i / (float)(n - 1) * 255.0f));

    // Composite with a head-to-tail gradient. The anterior is paler and cooler
    // (the pharynx catching light), deepening to the body green behind it,
    // which is what makes the eating end obvious on a screen this small without
    // gluing a separate highlight onto the nose. The flash lifts the whole
    // animal toward white as a word goes in.
    // In x-ray the body recedes so the connectome reads through it.
    uint32_t fm = (uint32_t)(c->flash * 95.0f) | (c->xray ? 0x10000u : 0u);
    if (fm != body_lut_key) {
        // One hue the whole way down. The previous ramp ran head colour into
        // body colour over just the front 18% of the animal, which is steep
        // enough that each capsule landed on a visibly different shade — that
        // was the "rainbow": per-capsule quantisation of a too-fast gradient,
        // not an intentional pattern.
        //
        // Now the pale anterior eases out over the first quarter, and the
        // banding is put back deliberately as a slow brightness wave along the
        // body — 13 bands, +-14%. C. elegans really is segmented by its
        // body-wall muscles, so it reads as anatomy rather than as an artefact.
        uint32_t f = fm & 0xFFFFu;
        uint32_t head_c = lerp_rgb(WR_HEAD, WR_FIRE, f);
        uint32_t body_c = lerp_rgb(WR_ACCENT, WR_FIRE, f);
        if (c->xray) {
            head_c = lerp_rgb(0x000000, head_c, 78);
            body_c = lerp_rgb(0x000000, body_c, 78);
        }
        for (int m = 0; m < 256; m++) {
            float t = (float)m / 255.0f;
            float head_mix = 1.0f - t / 0.24f;
            if (head_mix < 0.0f) head_mix = 0.0f;
            head_mix *= head_mix;  // ease, so the transition has no visible edge
            uint32_t base = lerp_rgb(body_c, head_c, (uint32_t)(head_mix * 255.0f));
            float band = 0.86f + 0.14f * sinf(t * 13.0f * 2.0f * (float)M_PI);
            uint32_t r = (uint32_t)(((base >> 16) & 0xFF) * band);
            uint32_t g = (uint32_t)(((base >> 8) & 0xFF) * band);
            uint32_t b = (uint32_t)((base & 0xFF) * band);
            body_lut[m] = to565((r > 255 ? 255 : r) << 16 | (g > 255 ? 255 : g) << 8 |
                                (b > 255 ? 255 : b));
        }
        body_lut_key = fm;
    }

    for (int y = 0; y < h; y++) {
        uint8_t *crow = c->cov + (size_t)y * WR_W;
        uint8_t *srow = c->seg + (size_t)y * WR_W;
        uint16_t *drow = c->band + (size_t)y * WR_W;
        for (int x = 0; x < WR_W; x++) {
            if (!crow[x]) continue;
            uint32_t m = srow[x];  // position along the body, 0 head .. 255 tail
            uint32_t a = crow[x];
            if (a >= 255) {
                drow[x] = body_lut[m];  // the interior, which is most of it
            } else {
                uint32_t src = body_lut[m], d = drow[x], ia = 255 - a;
                drow[x] = (uint16_t)(
                    (((((src >> 11) & 0x1F) * a + ((d >> 11) & 0x1F) * ia) * 257 >> 16) << 11) |
                    (((((src >> 5) & 0x3F) * a + ((d >> 5) & 0x3F) * ia) * 257 >> 16) << 5) |
                    ((((src & 0x1F) * a + (d & 0x1F) * ia) * 257 >> 16)));
            }
            c->cov_pixels++;
        }
    }

    // Neurons. Every eighth segment carries a node; they light up along the
    // body while the flash lasts, so eating reads as something travelling
    // backward through the animal rather than a light being switched on.
    if (c->flash > 0.02f) {
        for (int i = 4; i < n - 1; i += 8) {
            float t = (float)i / (float)(n - 1);
            // Wave sweeping head to tail over the life of the flash.
            float phase = (1.0f - c->flash) * 1.6f - t;
            float lit = 1.0f - fabsf(phase) * 2.2f;
            if (lit <= 0.0f) continue;
            uint32_t a = (uint32_t)(lit * c->flash * 230.0f);
            if (a > 255) a = 255;
            dot(c, y0, h, px[i], py[i], rad[i] * 0.52f + 1.2f, WR_FIRE, a);
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
    c->frame++;
    c->cam_x = (float)w->body.target_x;
    c->cam_y = (float)w->body.target_y;
    if (c->shudder > 0.001f) {
        // Two incommensurate frequencies so it reads as a jitter rather than a
        // wobble. Camera only — the body itself is the simulation's business.
        float t = (float)c->frame;
        c->cam_x += c->shudder * 7.0f * sinf(t * 2.7f) / scale_of(c);
        c->cam_y += c->shudder * 7.0f * sinf(t * 3.9f + 1.3f) / scale_of(c);
    }
    c->cov_pixels = 0;

    // Project the midline once for the whole frame. It is 201 points, and
    // reprojecting it per band would repeat that work for every one of them.
    // midline() is the head point followed by every segment tail.
    static float px[WM_N_SEGMENTS + 1], py[WM_N_SEGMENTS + 1];
    const wm_body *b = &w->body;
    w2s(c, (float)b->hx[0], (float)b->hy[0], &px[0], &py[0]);
    for (int i = 0; i < WM_N_SEGMENTS; i++)
        w2s(c, (float)b->tx[i], (float)b->ty[i], &px[i + 1], &py[i + 1]);

    // Per-segment radius, once for the whole frame: body_profile() calls powf
    // and sinf, and running it per band meant 3000 transcendental calls for 200
    // distinct answers.
    static float rad[WM_N_SEGMENTS];
    float sc = scale_of(c);
    for (int i = 0; i < WM_N_SEGMENTS; i++)
        rad[i] = c->body_radius * body_profile((float)i / (float)WM_N_SEGMENTS) * sc;

    // Project the frame once, then cull to the viewport once. The ellipsoid
    // spans the whole 1600x1060 world while the camera shows 400 units of it,
    // so all but a handful of its ~1250 segments are off screen every frame.
    // Testing them all against all fifteen bands was costing more than the
    // lines themselves.
    static float fx[WR_FRAME_MAX_PTS], fy[WR_FRAME_MAX_PTS];
    for (int i = 0; i < wr_frame_n; i++)
        w2s(c, wr_frame[i].wx, wr_frame[i].wy, &fx[i], &fy[i]);

    static uint16_t vis[WR_FRAME_MAX_PTS];
    static int16_t vis_y0[WR_FRAME_MAX_PTS], vis_y1[WR_FRAME_MAX_PTS];
    int n_vis = 0;
    const float m = WR_FRAME_HALF_WIDTH + 1.0f;
    for (int i = 1; i < wr_frame_n; i++) {
        if (wr_frame[i].start) continue;  // pen up between polylines
        float x0 = fminf(fx[i - 1], fx[i]) - m, x1 = fmaxf(fx[i - 1], fx[i]) + m;
        if (x1 < 0.0f || x0 >= (float)WR_W) continue;
        float ya = fminf(fy[i - 1], fy[i]) - m, yb = fmaxf(fy[i - 1], fy[i]) + m;
        if (yb < 0.0f || ya >= (float)WR_H) continue;
        vis[n_vis] = (uint16_t)i;
        vis_y0[n_vis] = (int16_t)ya;
        vis_y1[n_vis] = (int16_t)yb;
        n_vis++;
    }

    const uint16_t stage = to565(WR_STAGE);
    float ring_r = (1.0f - c->flash) * 210.0f + 14.0f;
    uint32_t ring_a = (uint32_t)(c->flash * c->flash * 150.0f);

    for (int y0 = 0; y0 < WR_H; y0 += c->band_rows) {
        int h = WR_H - y0 < c->band_rows ? WR_H - y0 : c->band_rows;
        size_t n = (size_t)WR_W * h;

        c->band = c->band_mem + (size_t)c->band_parity * WR_W * c->band_rows;
        c->band_parity ^= 1;

        for (size_t i = 0; i < n; i++) c->band[i] = stage;

        uint32_t t0 = WR_T();
        for (int k = 0; k < n_vis; k++) {
            if (vis_y1[k] < y0 || vis_y0[k] >= y0 + h) continue;
            int i = vis[k];
            frame_line(c, y0, h, fx[i - 1], fy[i - 1], fx[i], fy[i],
                       WR_FRAME_HALF_WIDTH, wr_frame[i].alpha);
        }

        uint32_t t1 = WR_T();
        band_words(c, w, y0, h);
        uint32_t t2 = WR_T();
        band_worm(c, y0, h, px, py, rad);
        uint32_t t3 = WR_T();
        wr_us_frame += t1 - t0; wr_us_words += t2 - t1; wr_us_worm += t3 - t2;
        if (c->xray) band_xray(c, w, y0, h, px, py, rad);
        if (ring_a > 4) pulse_ring(c, y0, h, ring_r, ring_a);

        if (c->title_alpha > 0.01f && c->title) {
            // Veil the scene behind the card. Without it the worm and the words
            // fight the label for the same pixels; with it the piece simply
            // fades up out of black as the name fades out.
            uint32_t veil = (uint32_t)(c->title_alpha * 205.0f);
            for (size_t i = 0; i < n; i++) blend(&c->band[i], 0x000000, veil);

            // Letterspaced, because a name set wide reads as a label rather
            // than as another word in the field the worm might eat.
            uint32_t a = (uint32_t)(c->title_alpha * 255.0f);
            draw_text_t(c, y0, h, c->title, (int)strlen(c->title),
                        // Low on the panel, clear of the worm: the camera
                        // rides the head, so screen centre is always occupied.
                        WR_W * 0.5f, WR_H * 0.5f - 15.0f, WR_HEAD, a, 6.0f);
            if (c->subtitle)
                draw_text_t(c, y0, h, c->subtitle, (int)strlen(c->subtitle),
                            WR_W * 0.5f, WR_H * 0.5f + 20.0f, WR_ACCENT,
                            a * 170 / 255, 2.0f);
        }
        // Before the mask, so the corners outside the panel stay black
        // instead of flashing white.
        if (c->invert)
            for (size_t i = 0; i < n; i++) c->band[i] = (uint16_t)~c->band[i];

        if (c->round_mask) band_mask(c, y0, h);

        blit(user, y0, h, c->band);
    }
}
