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
// Draw order, back to front: the alien sphere and the word haze (one coarse
// field, resampled up), the globe graticule, the words, the stippled worm, the
// flash.
//
// Two things are computed on a 64x64 grid rather than per pixel — the sphere and
// the clouds behind the words — because both are genuinely low-frequency. That
// turns a full-screen shader into 4096 evaluations plus a resample, which is the
// only reason any of this fits in the frame budget.

#include "wormrender.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Per-stage microseconds, for tuning. Costs nothing while wr_clock is NULL.
uint32_t wr_us_sphere, wr_us_bg, wr_us_frame, wr_us_words, wr_us_worm;
uint32_t (*wr_clock)(void);
#define WR_T() (wr_clock ? wr_clock() : 0u)

// viewer/focus/worm-render.js uses WORM_BASE_RADIUS = 22 world units, but its
// camera frames the whole 800-unit body. Here the window is 400 units, so 22
// reads as a snake. 15 keeps the nematode slenderness at this magnification.
#define WR_DEFAULT_RADIUS 15.0f

// How far the head may get from the middle of the panel, px.
#define WR_CAM_LEASH 105.0f

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

static void build_stipple(void);
static void build_bg_tables(void);
static void bg_attach(uint8_t *mem);

void wr_init(wr_ctx *c, uint8_t *scratch, int band_rows, uint8_t *bg) {
    memset(c, 0, sizeof(*c));
    c->band_rows = band_rows;
    c->band_mem = (uint16_t *)scratch;
    c->band = c->band_mem;
    c->band_parity = 0;
    c->cov = scratch + (size_t)WR_W * band_rows * 4;
    c->seg = c->cov + (size_t)WR_W * band_rows;
    bg_attach(bg);

    // 400 world units across = 2 x FOOD_SENSE_RADIUS, so the rim of the round
    // display sits exactly on the edge of what the worm can smell.
    c->view_units = 400.0f;
    c->body_radius = WR_DEFAULT_RADIUS;
    c->round_mask = true;
    // The graticule is set dressing now rather than the background itself.
    c->globe_alpha = 0.22f;
    c->bg_alien = true;
    c->haze = 1.0f;
    c->stipple = 1.0f;
    c->cam_lag = 2.5f;
    c->anatomy = true;

    build_stipple();
    build_bg_tables();
}

static inline float scale_of(const wr_ctx *c) { return (float)WR_W / c->view_units; }

// World y grows downward on screen: the scroller spawns words at y = 980 and
// decreases y to move them up the display (text_scroller.py SCROLL_SPEED).
static inline void w2s(const wr_ctx *c, float wx, float wy, float *sx, float *sy) {
    float s = scale_of(c);
    *sx = (wx - c->cam_x) * s + WR_W / 2.0f;
    *sy = (wy - c->cam_y) * s + WR_H / 2.0f;
}

// --- the background -----------------------------------------------------------
//
// An orthographic ball with three-dimensional value noise on its surface, the
// noise's own domain warped by a coarser octave of itself so the pattern folds
// inward as the ball turns. It is the one thing on the panel that is neither the
// animal nor the play, and it is kept dark on purpose: on an AMOLED the black
// around it is off, and a background bright enough to light the whole panel
// would throw away the depth that buys.
//
// It is evaluated on a WR_BG_N x WR_BG_N grid — 4096 samples instead of 217156 —
// and bilinearly resampled on the way to the panel. Nothing in it has detail
// finer than seven pixels, so nothing is lost, and the same field carries the
// haze the words sit in.

// Radius of the ball as a fraction of the half-screen. Slightly over 1 so its
// limb sits just outside the round panel and the darkness at the edge is the
// sphere's own falloff rather than a visible horizon.
#define WR_BG_SPHERE 1.06f
#define WR_BG_GAIN 0.44f  // ceiling on the sphere's brightness
#define WR_HAZE_CORE 1.42f  // 1/(1 - the fraction of the ellipse that stays flat)
#define WR_HAZE_MAX 186.0f  // how white a word's cloud gets over the word itself
#define WR_DUST_THRESH 16  // stipple threshold below which a background dot lights
#define WR_DUST_ALPHA 40

// Recomputing the noise every third frame. The ball turns at 0.11 rad/s and the
// field is a 7 px-per-cell blur; at 12-20 fps there is nothing in it that moves
// far enough in three frames to see. The haze is refreshed every frame
// regardless — that one tracks the words.
#define WR_BG_NOISE_EVERY 3

#define WR_BG_BYTES ((size_t)WR_BG_N * WR_BG_N * 3)

static uint8_t *bg_noise;  // the ball alone
static uint8_t *bg_field;  // the ball plus this frame's haze
static uint32_t bg_noise_frame = 0xFFFFFFFFu;

size_t wr_bg_bytes(void) { return WR_BG_BYTES * 2; }

static void bg_attach(uint8_t *mem) {
    bg_noise = mem;
    bg_field = mem + WR_BG_BYTES;
}

// Everything in the field dies before the bezel, so neither the ball nor a
// word's cloud gets sliced off square by the round mask. Computed rather than
// tabulated: it is a handful of operations and the table was 4 KB.
static inline float bg_rim_at(int i, int j) {
    float u = ((float)i + 0.5f) / WR_BG_N * 2.0f - 1.0f;
    float v = ((float)j + 0.5f) / WR_BG_N * 2.0f - 1.0f;
    float k = (1.02f - sqrtf(u * u + v * v)) * 5.0f;
    if (k < 0.0f) k = 0.0f;
    if (k > 1.0f) k = 1.0f;
    return k * k * (3.0f - 2.0f * k);
}

// Resample tables: for each output pixel, which pair of cells it lies between
// and how far along. Built once — the mapping is fixed by the geometry.
static uint16_t bg_ix[WR_W], bg_iy[WR_H];
static uint8_t bg_fx[WR_W], bg_fy[WR_H];

// Two horizontally-resampled rows, held across the bands of a frame, already in
// the panel's own RGB565. Output rows arrive in order, so a source row is
// expanded once and then reused by the seven or so output rows inside it — and
// the remaining per-pixel work is one blend between two packed pixels, which is
// four multiplies rather than three lerps and a pack.
//
// Interpolating in 565 rather than in 8-bit costs nothing real: the panel is
// 565, so both paths land on the same set of colours.
static uint16_t bg_rowbuf[2][WR_W];
static uint16_t *bg_rowA = bg_rowbuf[0], *bg_rowB = bg_rowbuf[1];
static int bg_row_have = -1;  // source row currently in bg_rowA

// One 32-bit mix. Not a good hash in any cryptographic sense; it only has to
// decorrelate three small integers, and it has to be cheap, because the noise
// asks for eight of them per octave.
static inline uint32_t hashi(int32_t x, int32_t y, int32_t z) {
    uint32_t h = (uint32_t)x * 0x1F123BB5u ^ (uint32_t)y * 0x85EBCA6Bu ^
                 (uint32_t)z * 0xC2B2AE35u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 13;
    return h;
}

static inline float hashf(int32_t x, int32_t y, int32_t z) {
    return (float)(hashi(x, y, z) >> 9) * (1.0f / 8388608.0f);
}

static float vnoise3(float x, float y, float z) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = x - fx, ty = y - fy, tz = z - fz;
    // Smoothstep the interpolants, not the values: trilinear alone leaves the
    // cell lattice visible as creases, which on a slowly turning ball reads as a
    // wireframe rather than as matter.
    tx = tx * tx * (3.0f - 2.0f * tx);
    ty = ty * ty * (3.0f - 2.0f * ty);
    tz = tz * tz * (3.0f - 2.0f * tz);

    float c000 = hashf(ix, iy, iz), c100 = hashf(ix + 1, iy, iz);
    float c010 = hashf(ix, iy + 1, iz), c110 = hashf(ix + 1, iy + 1, iz);
    float c001 = hashf(ix, iy, iz + 1), c101 = hashf(ix + 1, iy, iz + 1);
    float c011 = hashf(ix, iy + 1, iz + 1), c111 = hashf(ix + 1, iy + 1, iz + 1);

    float x00 = c000 + (c100 - c000) * tx, x10 = c010 + (c110 - c010) * tx;
    float x01 = c001 + (c101 - c001) * tx, x11 = c011 + (c111 - c011) * tx;
    float y0 = x00 + (x10 - x00) * ty, y1 = x01 + (x11 - x01) * ty;
    return y0 + (y1 - y0) * tz;
}

static float fbm3(float x, float y, float z, int octaves) {
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        sum += amp * vnoise3(x, y, z);
        norm += amp;
        amp *= 0.5f;
        x *= 2.03f;  // not exactly 2: an integer lacunarity lines the octaves'
        y *= 2.03f;  // lattices up and the creases come back
        z *= 2.03f;
    }
    return sum / norm;
}

static void build_bg_tables(void) {
    for (int x = 0; x < WR_W; x++) {
        float u = ((float)x + 0.5f) * (float)WR_BG_N / (float)WR_W - 0.5f;
        int i = (int)floorf(u);
        float f = u - (float)i;
        if (i < 0) { i = 0; f = 0.0f; }
        if (i > WR_BG_N - 2) { i = WR_BG_N - 2; f = 1.0f; }
        bg_ix[x] = (uint16_t)i;
        bg_fx[x] = (uint8_t)(f * 255.0f + 0.5f);
    }
    for (int y = 0; y < WR_H; y++) {
        float v = ((float)y + 0.5f) * (float)WR_BG_N / (float)WR_H - 0.5f;
        int j = (int)floorf(v);
        float f = v - (float)j;
        if (j < 0) { j = 0; f = 0.0f; }
        if (j > WR_BG_N - 2) { j = WR_BG_N - 2; f = 1.0f; }
        bg_iy[y] = (uint16_t)j;
        bg_fy[y] = (uint8_t)(f * 255.0f + 0.5f);
    }
    bg_row_have = -1;
    bg_noise_frame = 0xFFFFFFFFu;
}

static void build_bg_noise(float t) {
    // The ball turns about an axis that is itself tilted, so no feature ever
    // traces the same path twice and the poles stay off the silhouette.
    const float ca = cosf(t * 0.11f), sa = sinf(t * 0.11f);
    const float ct = cosf(0.42f), st = sinf(0.42f);
    // A fixed key light, high and to the left, so the thing reads as a solid
    // rather than as a disc of texture.
    const float lx = -0.42f, ly = -0.52f, lz = 0.74f;

    for (int j = 0; j < WR_BG_N; j++) {
        for (int i = 0; i < WR_BG_N; i++) {
            uint8_t *o = &bg_noise[((size_t)j * WR_BG_N + i) * 3];
            float u = (((float)i + 0.5f) / WR_BG_N * 2.0f - 1.0f) / WR_BG_SPHERE;
            float v = (((float)j + 0.5f) / WR_BG_N * 2.0f - 1.0f) / WR_BG_SPHERE;
            float r2 = u * u + v * v;
            if (r2 >= 1.0f) { o[0] = o[1] = o[2] = 0; continue; }
            float z = sqrtf(1.0f - r2);

            float px = u * ca + z * sa, pz = -u * sa + z * ca, py = v;
            float qy = py * ct - pz * st, qz = py * st + pz * ct;

            // Domain warp: a coarse field displacing a finer one. This is what
            // makes it fold rather than merely drift — the fine pattern is being
            // dragged through itself.
            float wv = fbm3(px * 1.5f, qy * 1.5f, qz * 1.5f + t * 0.07f, 2);
            float n = fbm3(px * 4.3f + wv * 3.4f, qy * 4.3f + wv * 3.4f,
                           qz * 4.3f - t * 0.05f, 3);

            // Ridged: fold the field about its midpoint so the level set becomes
            // a crest instead of a plateau. Plain fbm on a sphere reads as
            // smoke; the ridges read as something with structure inside it,
            // which is the whole point of putting it there.
            float sh = 1.0f - fabsf(n * 2.0f - 1.0f);
            sh = sh * sh;
            // Then push the midtones down: only the crests glow, and the rest
            // stays close enough to black that the panel's own black still reads
            // as depth.
            sh = sh * 1.55f - 0.42f;
            if (sh < 0.0f) sh = 0.0f;
            if (sh > 1.0f) sh = 1.0f;

            float lam = u * lx + v * ly + z * lz;
            if (lam < 0.0f) lam = 0.0f;
            float lit = (0.18f + 0.82f * lam) * WR_BG_GAIN;

            uint32_t col = sh < 0.5f
                ? lerp_rgb(WR_BG_LOW, WR_BG_MID, (uint32_t)(sh * 510.0f))
                : lerp_rgb(WR_BG_MID, WR_BG_HIGH, (uint32_t)((sh - 0.5f) * 510.0f));

            // A cold fresnel at the limb. Cheap sphere-ness: the eye reads a
            // bright edge on a dark disc as curvature without being told.
            float fr = 1.0f - z;
            fr = fr * fr * fr * fr;

            float rr = (float)((col >> 16) & 0xFF) * lit + ((WR_BG_RIM >> 16) & 0xFF) * fr;
            float gg = (float)((col >> 8) & 0xFF) * lit + ((WR_BG_RIM >> 8) & 0xFF) * fr;
            float bb = (float)(col & 0xFF) * lit + (WR_BG_RIM & 0xFF) * fr;

            float k = bg_rim_at(i, j);
            uint32_t ir = (uint32_t)(rr * k), ig = (uint32_t)(gg * k),
                     ib = (uint32_t)(bb * k);
            o[0] = (uint8_t)(ir > 255 ? 255 : ir);
            o[1] = (uint8_t)(ig > 255 ? 255 : ig);
            o[2] = (uint8_t)(ib > 255 ? 255 : ib);
        }
    }
}

static void bg_expand_row(int src, uint16_t *out) {
    const uint8_t *row = &bg_field[(size_t)src * WR_BG_N * 3];
    for (int x = 0; x < WR_W; x++) {
        const uint8_t *a = row + (size_t)bg_ix[x] * 3;
        int f = bg_fx[x];
        uint32_t r = (uint32_t)(a[0] + (((int)a[3] - (int)a[0]) * f >> 8));
        uint32_t g = (uint32_t)(a[1] + (((int)a[4] - (int)a[1]) * f >> 8));
        uint32_t b = (uint32_t)(a[2] + (((int)a[5] - (int)a[2]) * f >> 8));
        out[x] = (uint16_t)((r >> 3) << 11 | (g >> 2) << 5 | (b >> 3));
    }
}

// Fill a band from the coarse field, and sprinkle the background dot lattice
// while every pixel is already in hand. The dust is the same stipple screen the
// worm is drawn through, at its lowest threshold — one pixel per dot cell — so
// the animal reads as those dots thickening rather than as a separate object
// laid over them.
static uint8_t stipple[WR_STIPPLE_TILE * WR_STIPPLE_TILE];
static uint8_t stip_gain[256];  // how bright a dot is, by the coverage under it
static uint8_t stip_veil[256];  // and how far the ground under it is pushed down
static uint8_t dust_x[WR_STIPPLE_TILE][12];  // where the background dots fall
static uint8_t dust_n[WR_STIPPLE_TILE];

static void band_bg(wr_ctx *c, int y0, int h) {
    // A fifth of every band is corner that the round mask will throw away, and
    // this is the costliest per-pixel loop in the renderer. Skip it here; the
    // mask still runs at the end, because the worm and the graticule can reach
    // out there afterwards.
    const float mcx = WR_W / 2.0f, mcy = WR_H / 2.0f, mr = WR_W / 2.0f - 0.5f;
    for (int y = 0; y < h; y++) {
        int sy = y0 + y;
        int r0 = bg_iy[sy];
        if (r0 != bg_row_have) {
            if (r0 == bg_row_have + 1) {
                uint16_t *t = bg_rowA;
                bg_rowA = bg_rowB;
                bg_rowB = t;
                bg_expand_row(r0 + 1, bg_rowB);
            } else {
                bg_expand_row(r0, bg_rowA);
                bg_expand_row(r0 + 1, bg_rowB);
            }
            bg_row_have = r0;
        }
        uint32_t t = bg_fy[sy] >> 3, it = 32 - t;  // 5 bits is all 565 can use
        const uint16_t *A = bg_rowA, *B = bg_rowB;
        uint16_t *dst = c->band + (size_t)y * WR_W;

        float dy = (float)sy + 0.5f - mcy;
        float q = mr * mr - dy * dy;
        if (q <= 0.0f) {
            memset(dst, 0, sizeof(uint16_t) * WR_W);
            continue;
        }
        float hx = sqrtf(q);
        int x0 = (int)(mcx - hx), x1 = (int)(mcx + hx);
        if (x0 < 0) x0 = 0;
        if (x1 >= WR_W) x1 = WR_W - 1;
        memset(dst, 0, sizeof(uint16_t) * (size_t)x0);
        memset(dst + x1 + 1, 0, sizeof(uint16_t) * (size_t)(WR_W - 1 - x1));

        for (int x = x0; x <= x1; x++) {
            uint32_t a = A[x], b = B[x];
            uint32_t rb = ((a & 0xF81Fu) * it + (b & 0xF81Fu) * t) >> 5;
            uint32_t g = ((a & 0x07E0u) * it + (b & 0x07E0u) * t) >> 5;
            dst[x] = (uint16_t)((rb & 0xF81Fu) | (g & 0x07E0u));
        }

        // The dust, from a list rather than a test. Its positions are fixed by
        // the stipple screen, so the pixels that light are known ahead of time —
        // about fourteen a row, against 466 comparisons if it were a per-pixel
        // branch inside the loop above.
        const uint8_t *dx = dust_x[sy & (WR_STIPPLE_TILE - 1)];
        int dn = dust_n[sy & (WR_STIPPLE_TILE - 1)];
        for (int base = 0; base < WR_W; base += WR_STIPPLE_TILE)
            for (int k = 0; k < dn; k++) {
                int x = base + dx[k];
                if (x < x0 || x > x1) continue;
                blend(dst + x, WR_DUST, WR_DUST_ALPHA);
            }
    }
}

// --- the stipple screen -------------------------------------------------------
//
// A clustered-dot threshold field on a jittered lattice, tiled across the panel
// and fixed to it. A pixel lights when the body's coverage there exceeds its
// threshold, and the threshold rises as the square of the distance to the
// nearest dot centre — so a dot's area is linear in coverage and the animal
// arriving underneath makes the dots swell rather than makes new ones appear.
//
// Screen-fixed, not body-fixed: the lattice does not move, so the worm reads as
// something passing under a fabric rather than as something wearing a texture.

#define WR_STIPPLE_CELLS 11  // dot centres per tile edge: 64/11 = 5.8 px apart
#define WR_STIPPLE_R 0.50f   // max dot radius, in lattice spacings
#define WR_STIPPLE_GAIN 242  // of 256: the ceiling that keeps the interstices open
#define WR_STIPPLE_EDGE 5    // how hard a dot's own edge is
#define WR_BODY_BANDS 0.0f   // amplitude of the old brightness wave; 0.14 restores it
#define WR_BODY_VEIL 0.72f   // how far the ground under the animal is pushed down

static void build_stipple(void) {
    const int C = WR_STIPPLE_CELLS, T = WR_STIPPLE_TILE;
    const float pitch = (float)T / (float)C;
    const float rmax = pitch * WR_STIPPLE_R;
    static float cx[WR_STIPPLE_CELLS * WR_STIPPLE_CELLS];
    static float cy[WR_STIPPLE_CELLS * WR_STIPPLE_CELLS];
    static float cr[WR_STIPPLE_CELLS * WR_STIPPLE_CELLS];

    for (int j = 0; j < C; j++) {
        for (int i = 0; i < C; i++) {
            uint32_t hh = hashi(i, j, 17);
            // Half-offset odd rows for a hex-ish packing, then jitter, so the
            // screen does not read as the square grid it is built on.
            float ox = (j & 1) ? 0.5f : 0.0f;
            cx[j * C + i] = ((float)i + 0.5f + ox + ((float)(hh & 255) / 255.0f - 0.5f) * 0.30f) * pitch;
            cy[j * C + i] = ((float)j + 0.5f + ((float)((hh >> 8) & 255) / 255.0f - 0.5f) * 0.30f) * pitch;
            cr[j * C + i] = rmax * (0.86f + 0.30f * (float)((hh >> 16) & 255) / 255.0f);
        }
    }

    for (int y = 0; y < T; y++) {
        for (int x = 0; x < T; x++) {
            float best = 1.0f;
            for (int k = 0; k < C * C; k++) {
                float dx = fabsf((float)x + 0.5f - cx[k]);
                float dy = fabsf((float)y + 0.5f - cy[k]);
                if (dx > T * 0.5f) dx = (float)T - dx;  // the tile wraps
                if (dy > T * 0.5f) dy = (float)T - dy;
                float d2 = dx * dx + dy * dy;
                float q = d2 / (cr[k] * cr[k]);
                if (q < best) best = q;
            }
            stipple[y * T + x] = (uint8_t)(best * 255.0f);
        }
    }

    for (int y = 0; y < T; y++) {
        int n = 0;
        for (int x = 0; x < T && n < (int)sizeof(dust_x[0]); x++)
            if (stipple[y * T + x] < WR_DUST_THRESH) dust_x[y][n++] = (uint8_t)x;
        dust_n[y] = (uint8_t)n;
    }

    for (int i = 0; i < 256; i++) {
        float t = (float)i / 240.0f;
        if (t > 1.0f) t = 1.0f;
        stip_gain[i] = (uint8_t)(70.0f + 185.0f * t * t);
        // The animal darkens what it passes over, in 32nds. Without this the
        // green dots land on a white word-cloud at nearly the same value as the
        // black type beside them and the two read as one texture; with it the
        // worm always has its own dark ground and sits unmistakably in front.
        // The words stay legible through it, which is the point — it is a
        // shadow, not an eraser.
        float v = (float)i / 255.0f;
        stip_veil[i] = (uint8_t)(32.0f - 32.0f * WR_BODY_VEIL * v * v);
    }
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

void wr_build_globe(void) {
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


// Depth. A word does not switch on when it crosses the rim and off when it
// leaves; it arrives, the way something entering the edge of your attention
// does. Over the bottom two fifths of the panel it grows from a bit over half
// size and fades up, holds through the middle, then over the top third fades out
// and draws back down again — which reads as passing overhead rather than as
// being deleted.
//
// The distances are large on purpose. Words climb at 15 world units a second,
// about 17 px, so the arrival takes some thirteen seconds and the departure ten.
#define WR_FADE_IN_BOT 1.05f   // fraction of panel height where a word starts to exist
#define WR_FADE_IN_TOP 0.55f   // and where it is fully here
#define WR_FADE_OUT_TOP (-0.05f)  // where it has finished leaving
#define WR_FADE_OUT_BOT 0.38f  // and where leaving begins
#define WR_SCALE_MIN 0.54f     // its size the moment it appears at the bottom
#define WR_SCALE_OUT 0.86f     // and the size it has shrunk to as it goes

// The panel is a circle, so the vertical profile alone is not enough: a word
// off to one side crosses the bezel on an arc, not through the bottom, and used
// to switch on the moment it cleared the mask. This is the arrival that actually
// does most of the work — full weight inside 60% of the radius, nothing at the
// bezel, which is 103 px of travel and about four seconds at the speed the
// camera moves. It is also the peripheral-vision reading the framing already
// claims: what is off to the side is hazy, what is in the middle is sharp.
#define WR_FADE_RIM 1.06f
#define WR_FADE_CORE 0.60f

// A shadow needs something to fall on, which is what the haze is for. Offset
// down and right from a light that is up and left — the same one lighting the
// ball behind it. It is violet-grey rather than black, because the word itself
// is nearly black now: a black shadow under black type is just a heavier letter,
// and what should read is the sliver of it that falls past the stroke onto the
// cloud.
#define WR_SHADOW_DX 3.0f
#define WR_SHADOW_DY 4.0f
#define WR_SHADOW_A 0.55f

static inline float smoothstep_e(float e0, float e1, float x) {
    float t = (x - e0) / (e1 - e0);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

// How much of a glyph survives right at the bezel. The word as a whole has
// already been faded by its distance from the middle; this is only here so that
// the far end of a long word does not get cut off square by the round mask.
static inline float rim_fade(float x, float y) {
    float dx = x - WR_W * 0.5f, dy = y - WR_H * 0.5f;
    float r = sqrtf(dx * dx + dy * dy) / (WR_W * 0.5f);
    return smoothstep_e(1.06f, 0.94f, r);
}

// Centred on (cx, cy), matching the site's textAlign/textBaseline centre in
// viewer/focus/text-canvas.js. Rows outside the band are skipped.
//
// `scale` under 1 resamples the atlas bilinearly rather than picking a smaller
// baked size: the growth has to be continuous, and a word stepping between two
// baked sizes changes width by several pixels at once, which is visible as a
// twitch even under a fade. Full size takes the straight blit — that is where
// the words spend most of their life.
static void draw_glyphs(wr_ctx *c, int y0, int h, const char *s, int len,
                        float cx, float cy, uint32_t color, uint32_t alpha,
                        float tracking, float scale, bool rim) {
    if (alpha == 0) return;
    const bool exact = scale > 0.995f;
    float pen = cx - text_width_t(s, len, tracking) * scale / 2.0f;
    float base = cy + wr_font_size * 0.36f * scale;
    const float inv = 1.0f / scale;

    for (int i = 0; i < len;) {
        int used, gi = glyph_slot(s + i, len - i, &used);
        i += used;
        if (gi < 0) continue;
        const wr_glyph *g = &wr_font_glyph[gi];
        if (g->w) {
            int dw = exact ? g->w : (int)(g->w * scale + 0.5f);
            int dh = exact ? g->h : (int)(g->h * scale + 0.5f);
            if (dw < 1) dw = 1;
            if (dh < 1) dh = 1;
            int gx0 = (int)lroundf(pen + g->bx * scale);
            int gy0 = (int)lroundf(base + g->by * scale);

            uint32_t ga = alpha;
            if (rim) {
                float f = rim_fade((float)gx0 + dw * 0.5f, (float)gy0 + dh * 0.5f);
                ga = (uint32_t)(alpha * f);
                if (ga == 0) { pen += (g->adv + tracking) * scale; continue; }
            }

            if (exact) {
                for (int gy = 0; gy < g->h; gy++) {
                    int py = gy0 + gy - y0;
                    if (py < 0 || py >= h) continue;
                    const uint8_t *row =
                        wr_font_atlas + (size_t)(g->ay + gy) * wr_font_atlas_w + g->ax;
                    uint16_t *dst = c->band + (size_t)py * WR_W;
                    for (int gx = 0; gx < g->w; gx++) {
                        int px = gx0 + gx;
                        if (px < 0 || px >= WR_W) continue;
                        blend(dst + px, color, ((uint32_t)row[gx] * ga * 257) >> 16);
                    }
                }
            } else {
                for (int dy = 0; dy < dh; dy++) {
                    int py = gy0 + dy - y0;
                    if (py < 0 || py >= h) continue;
                    float fy = ((float)dy + 0.5f) * inv - 0.5f;
                    int sy0 = (int)floorf(fy);
                    uint32_t wy = (uint32_t)((fy - (float)sy0) * 256.0f);
                    if (sy0 < 0) { sy0 = 0; wy = 0; }
                    int sy1 = sy0 + 1;
                    if (sy1 > g->h - 1) { sy1 = g->h - 1; }
                    if (sy0 > g->h - 1) { sy0 = g->h - 1; wy = 0; }
                    const uint8_t *r0 =
                        wr_font_atlas + (size_t)(g->ay + sy0) * wr_font_atlas_w + g->ax;
                    const uint8_t *r1 =
                        wr_font_atlas + (size_t)(g->ay + sy1) * wr_font_atlas_w + g->ax;
                    uint16_t *dst = c->band + (size_t)py * WR_W;
                    for (int dx = 0; dx < dw; dx++) {
                        int px = gx0 + dx;
                        if (px < 0 || px >= WR_W) continue;
                        float fx = ((float)dx + 0.5f) * inv - 0.5f;
                        int sx0 = (int)floorf(fx);
                        uint32_t wx = (uint32_t)((fx - (float)sx0) * 256.0f);
                        if (sx0 < 0) { sx0 = 0; wx = 0; }
                        int sx1 = sx0 + 1;
                        if (sx1 > g->w - 1) sx1 = g->w - 1;
                        if (sx0 > g->w - 1) { sx0 = g->w - 1; wx = 0; }
                        uint32_t t0 = r0[sx0] * (256 - wx) + r0[sx1] * wx;
                        uint32_t t1 = r1[sx0] * (256 - wx) + r1[sx1] * wx;
                        uint32_t a = ((t0 * (256 - wy) + t1 * wy) >> 16);
                        blend(dst + px, color, (a * ga * 257) >> 16);
                    }
                }
            }
        }
        pen += (g->adv + tracking) * scale;
    }
}

static void draw_text_t(wr_ctx *c, int y0, int h, const char *s, int len,
                        float cx, float cy, uint32_t color, uint32_t alpha,
                        float tracking) {
    draw_glyphs(c, y0, h, s, len, cx, cy, color, alpha, tracking, 1.0f, false);
}

// The frame's visible words, resolved once in the prologue: screen position,
// depth, colour. Fifteen bands each re-deriving this per word was fifteen times
// the fade arithmetic and fifteen times the string lookup for one answer, and
// the haze pass needs the same list before any band is drawn.
typedef struct {
    const char *txt;
    uint16_t len;
    float sx, sy, scale;
    uint32_t rgb;
    uint16_t alpha;
    uint8_t lit;  // an edible word: carries a cloud and casts a shadow
} wr_wordv;

#define WR_MAX_VIS_WORDS 224
static wr_wordv wv[WR_MAX_VIS_WORDS];
static int wv_n;

static void build_words(wr_ctx *c, const wm_world *w) {
    const wm_asset *a = w->a;
    const wm_scroller *sc = &w->scroller;
    // Half the widest word in the corpus at this size, so a word is only
    // dropped once no part of it could reach the panel.
    const float margin = 160.0f;
    wv_n = 0;

    for (int li = 0; li < sc->n_lines && wv_n < WR_MAX_VIS_WORDS; li++) {
        const wm_line *L = &sc->lines[li];
        for (uint16_t wi = 0; wi < L->n_words && wv_n < WR_MAX_VIS_WORDS; wi++) {
            const wm_word *word = &L->w[wi];
            float sx, sy;
            w2s(c, (float)word->x, (float)word->y, &sx, &sy);
            if (sx < -margin || sx > WR_W + margin) continue;

            float tin = smoothstep_e(WR_H * WR_FADE_IN_BOT, WR_H * WR_FADE_IN_TOP, sy);
            float tout = smoothstep_e(WR_H * WR_FADE_OUT_TOP, WR_H * WR_FADE_OUT_BOT, sy);
            float ddx = sx - WR_W * 0.5f, ddy = sy - WR_H * 0.5f;
            float near = smoothstep_e(WR_FADE_RIM, WR_FADE_CORE,
                                      sqrtf(ddx * ddx + ddy * ddy) / (WR_W * 0.5f));
            float depth = near * tin * tout;
            if (depth <= 0.004f) continue;

            uint32_t color, alpha;
            bool lit = false;
            if (word->eaten) {
                color = WR_GHOST;
                alpha = 110;  // still there, but spent
            } else if (L->edible) {
                color = WR_WORD;
                alpha = 255;
                lit = true;
            } else {
                color = WR_WORD_INERT;
                alpha = 130;
            }

            uint32_t len;
            const char *txt = wm_str(&a->tok_text, word->tok, &len);
            wr_wordv *v = &wv[wv_n++];
            v->txt = txt;
            v->len = (uint16_t)len;
            v->sx = sx;
            v->sy = sy;
            v->scale = (WR_SCALE_MIN + (1.0f - WR_SCALE_MIN) * tin) *
                       (WR_SCALE_OUT + (1.0f - WR_SCALE_OUT) * tout);
            v->rgb = color;
            v->alpha = (uint16_t)(alpha * depth);
            v->lit = lit;
        }
    }
}

// Each live word puts a soft white bloom into the coarse background field. It
// costs a couple of hundred cell writes for the whole screen because a cloud has
// no detail worth resolving finer, and it comes out of the same resample as the
// sphere — so the words sit in weather rather than on top of black, and a black
// shadow has something to land on.
static void splat_haze(wr_ctx *c) {
    if (c->haze <= 0.01f) return;
    const float cell = (float)WR_W / (float)WR_BG_N;
    const float inv_cell = 1.0f / cell;

    for (int k = 0; k < wv_n; k++) {
        const wr_wordv *v = &wv[k];
        if (!v->lit || v->alpha < 6) continue;
        // An ellipse rather than a box: two 1-D profiles multiplied leave
        // corners, and a cloud with corners is a card.
        float rx = text_width_t(v->txt, v->len, 0.0f) * v->scale * 0.5f + 44.0f;
        float ry = (float)wr_font_size * v->scale * 0.62f + 34.0f;
        float inv_rx2 = 1.0f / (rx * rx), inv_ry2 = 1.0f / (ry * ry);
        // The cloud outlives the word slightly — ^0.7 rather than linear. Black
        // type has to keep something to be black against the whole way in and
        // the whole way out, and a cloud alone at the very end of that reads as
        // a word that has not arrived yet rather than as an error.
        float a01 = (float)v->alpha * (1.0f / 255.0f);
        float peak = WR_HAZE_MAX * c->haze * powf(a01, 0.7f);

        int i0 = (int)((v->sx - rx) * inv_cell);
        int i1 = (int)((v->sx + rx) * inv_cell) + 1;
        int j0 = (int)((v->sy - ry) * inv_cell);
        int j1 = (int)((v->sy + ry) * inv_cell) + 1;
        if (i0 < 0) i0 = 0;
        if (j0 < 0) j0 = 0;
        if (i1 > WR_BG_N - 1) i1 = WR_BG_N - 1;
        if (j1 > WR_BG_N - 1) j1 = WR_BG_N - 1;

        for (int j = j0; j <= j1; j++) {
            float dy = ((float)j + 0.5f) * cell - v->sy;
            float qy = dy * dy * inv_ry2;
            if (qy >= 1.0f) continue;
            for (int i = i0; i <= i1; i++) {
                float dx = ((float)i + 0.5f) * cell - v->sx;
                float q = qy + dx * dx * inv_rx2;
                if (q >= 1.0f) continue;
                // Flat over the word, falling only outside it. A peaked
                // profile leaves the ends of a long word on a dimmer cloud than
                // its middle, which black type shows immediately.
                float f = (1.0f - q) * WR_HAZE_CORE;
                if (f > 1.0f) f = 1.0f;
                f = f * f * (3.0f - 2.0f * f);

                size_t o = (size_t)j * WR_BG_N + i;
                uint32_t add = (uint32_t)(peak * f * bg_rim_at(i, j));
                if (!add) continue;
                uint8_t *px = &bg_field[o * 3];
                for (int ch = 0; ch < 3; ch++) {
                    uint32_t t = px[ch] + (add * ((WR_HAZE >> (16 - 8 * ch)) & 0xFF) >> 8);
                    px[ch] = (uint8_t)(t > 255 ? 255 : t);
                }
            }
        }
    }
}

static void band_words(wr_ctx *c, int y0, int h) {
    const float reach = (float)wr_font_size + 6.0f;
    for (int k = 0; k < wv_n; k++) {
        const wr_wordv *v = &wv[k];
        if (v->sy < y0 - reach || v->sy > y0 + h + reach) continue;
        if (v->lit)
            draw_glyphs(c, y0, h, v->txt, v->len, v->sx + WR_SHADOW_DX,
                        v->sy + WR_SHADOW_DY, WR_WORD_CAST,
                        (uint32_t)(v->alpha * WR_SHADOW_A), 0.0f, v->scale, true);
        draw_glyphs(c, y0, h, v->txt, v->len, v->sx, v->sy, v->rgb, v->alpha,
                    0.0f, v->scale, true);
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

// The fringe. Coverage carries on past the silhouette, falling to zero over
// WR_HALO px, and the stipple screen turns that ramp into dots that shrink with
// distance — which is where "the animal is the dot field thickening" actually
// happens. Without it the body is a stippled shape with a hard edge rather than
// something surfacing.
//
// Stamped as a disc every WR_HALO_STRIDE-th segment rather than as a capsule at
// every one. A halo is smooth by definition, and a 9 px annulus around a 5 px
// segment is almost entirely bounding box: at full density it cost more than the
// animal did. Discs of radius rad+9 every 20 px leave the envelope scalloped by
// about two pixels, which nothing rendered as dots can show.
//
// The falloff is quadratic in distance rather than linear because linear needs a
// square root, and the halo is the one part of the body that cannot use the
// saturation shortcut for its own pixels.
#define WR_HALO 9.0f
#define WR_HALO_TOP 254.0f
#define WR_HALO_STRIDE 4

static void cover_halo(wr_ctx *c, int y0, int h, float cx, float cy, float rad,
                       uint8_t seg) {
    if (rad < 0.35f) return;
    float ext = rad + WR_HALO;
    int miny = (int)floorf(cy - ext) - y0, maxy = (int)ceilf(cy + ext) - y0;
    if (maxy < 0 || miny >= h) return;
    if (miny < 0) miny = 0;
    if (maxy >= h) maxy = h - 1;

    float r_in2 = rad > 0.5f ? (rad - 0.5f) * (rad - 0.5f) : 0.0f;
    float r_halo2 = ext * ext;
    float halo_k = WR_HALO_TOP / (r_halo2 - r_in2);

    for (int py = miny; py <= maxy; py++) {
        float fy = (float)(py + y0) + 0.5f - cy;
        float q = r_halo2 - fy * fy;
        if (q <= 0.0f) continue;
        // The exact span, per row. A bounding box round a disc this fat is
        // three-quarters corner, and the corners are the pixels that cannot be
        // skipped cheaply.
        float hx = sqrtf(q);
        int minx = (int)(cx - hx), maxx = (int)(cx + hx);
        if (maxx < 0 || minx >= WR_W) continue;
        if (minx < 0) minx = 0;
        if (maxx >= WR_W) maxx = WR_W - 1;

        uint8_t *crow = c->cov + (size_t)py * WR_W;
        uint8_t *srow = c->seg + (size_t)py * WR_W;
        float fy2 = fy * fy;
        for (int px = minx; px <= maxx; px++) {
            if (crow[px] == 255) continue;  // the body proper, already drawn
            float ex = (float)px + 0.5f - cx;
            float d2 = ex * ex + fy2;
            uint8_t v = d2 <= r_in2 ? (uint8_t)WR_HALO_TOP
                                    : (uint8_t)((r_halo2 - d2) * halo_k);
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

// What is actually inside a nematode, at the only two scales this magnification
// can show: the pharynx and the gut.
//
// C. elegans is transparent. Under a microscope you do not see a green tube —
// you see the cuticle's outline and, straight through it, the muscular pharynx
// pumping at the front and the intestine running the rest of the length. Those
// two are most of what makes a photograph of one recognisable.
//
// The pharynx is the anterior sixth: an elongated corpus, a narrow isthmus, and
// a round terminal bulb. That silhouette — fat, thin, round — is the single most
// identifiable thing about the animal, and it costs about thirty discs along a
// midline that has already been projected for the body.
//
// Drawn after the body, so it reads as structure seen through the cuticle rather
// than as paint on top of it.

// s -> (radius as a fraction of the local body radius, colour, alpha).
// Segment indices, out of WM_N_SEGMENTS.
#define WR_PHARYNX_END 31
#define WR_GUT_START 38
#define WR_GUT_END 186

static void band_anatomy(wr_ctx *c, int y0, int h, const float *px, const float *py,
                         const float *rad) {
    // The gut: a darker cord at about a third of the body's width, from behind
    // the pharynx most of the way to the tail. Every segment, not every third —
    // segments project to 4.7 px apart and the cord is ~6 px across, so at a
    // stride of three it came out as a string of beads instead of an organ.
    for (int i = WR_GUT_START; i < WR_GUT_END && i < WM_N_SEGMENTS; i++) {
        float r = rad[i] * 0.32f;
        if (r < 0.6f) continue;
        // Fades out toward the tail, where the real intestine narrows and the
        // body is mostly gonad.
        float t = (float)(i - WR_GUT_START) / (float)(WR_GUT_END - WR_GUT_START);
        uint32_t a = (uint32_t)(165.0f * (1.0f - 0.55f * t));
        dot(c, y0, h, px[i], py[i], r, WR_GUT, a);
    }

    // The pharynx. Corpus, isthmus, terminal bulb — fat, thin, round.
    for (int i = 2; i <= WR_PHARYNX_END && i < WM_N_SEGMENTS; i++) {
        float t = (float)(i - 2) / (float)(WR_PHARYNX_END - 2);  // 0 at the mouth
        float w;
        if (t < 0.42f) {
            w = 0.60f - 0.10f * (t / 0.42f);        // corpus, tapering slightly
        } else if (t < 0.62f) {
            w = 0.24f;                               // isthmus
        } else {
            float u = (t - 0.62f) / 0.38f;           // terminal bulb
            w = 0.30f + 0.36f * sinf(u * (float)M_PI);
        }
        float r = rad[i] * w;
        if (r < 0.6f) continue;
        dot(c, y0, h, px[i], py[i], r, WR_PHARYNX, 150);
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

    // Then the fringe, over the body the core pass just saturated — so its own
    // interior pixels cost one compare each.
    if (c->stipple > 0.5f) {
        for (int i = 0; i < n - 1; i += WR_HALO_STRIDE)
            cover_halo(c, y0, h, px[i], py[i], rad[i],
                       (uint8_t)((float)i / (float)(n - 1) * 255.0f));
    }

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
        // The pale anterior eases out over the first quarter, and that is all
        // the variation there is. A slow brightness wave along the body used to
        // sit on top of it — 13 bands at +-14%, meant to read as the body-wall
        // muscles. Measured, it moved the hue by only 7 degrees, so it was never
        // the "rainbow" it was accused of being; what it actually did was put a
        // second light-and-dark texture on a body that already had one in the
        // stipple, and between them the shape stopped being readable. The real
        // internal structure is drawn as structure now — see band_anatomy — and
        // WR_BODY_BANDS is left at zero rather than deleted.
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
            float band = 1.0f - WR_BODY_BANDS +
                         WR_BODY_BANDS * sinf(t * 13.0f * 2.0f * (float)M_PI);
            uint32_t r = (uint32_t)(((base >> 16) & 0xFF) * band);
            uint32_t g = (uint32_t)(((base >> 8) & 0xFF) * band);
            uint32_t b = (uint32_t)((base & 0xFF) * band);
            body_lut[m] = to565((r > 255 ? 255 : r) << 16 | (g > 255 ? 255 : g) << 8 |
                                (b > 255 ? 255 : b));
        }
        body_lut_key = fm;
    }

    // Composite through the stipple screen. A pixel lights when the body's
    // coverage there beats its threshold, so a dot's area tracks coverage and
    // the animal resolves out of the lattice as it arrives underneath. The gain
    // is held under full scale on purpose: at 255 the dots would flood together
    // into the old solid body and there would be nothing to see through.
    const int T = WR_STIPPLE_TILE;
    const bool stip = c->stipple > 0.5f;
    for (int y = 0; y < h; y++) {
        uint8_t *crow = c->cov + (size_t)y * WR_W;
        uint8_t *srow = c->seg + (size_t)y * WR_W;
        uint16_t *drow = c->band + (size_t)y * WR_W;
        const uint8_t *trow = stipple + (size_t)((y + y0) & (T - 1)) * T;
        for (int x = 0; x < WR_W; x++) {
            uint32_t cv = crow[x];
            if (!cv) continue;
            uint32_t a;
            if (stip) {
                // The animal's own ground, laid down before its dots. Skipped
                // out in the thin part of the fringe, where there is nothing to
                // separate from anyway and the pixels are many.
                if (cv > 56) {
                    uint32_t vk = stip_veil[cv], d = drow[x];
                    drow[x] = (uint16_t)((((d & 0xF81Fu) * vk >> 5) & 0xF81Fu) |
                                         (((d & 0x07E0u) * vk >> 5) & 0x07E0u));
                }
                uint32_t thr = trow[x & (T - 1)];
                uint32_t v = (cv * WR_STIPPLE_GAIN) >> 8;
                if (v <= thr) continue;
                a = (v - thr) * WR_STIPPLE_EDGE;
                if (a > 255) a = 255;
                // Out in the halo the dots are not only smaller but fainter, so
                // the field thins into the background instead of ending.
                a = (a * stip_gain[cv]) >> 8;
                if (!a) continue;
            } else {
                a = cv;  // no halo pass ran, so this is plain edge coverage
            }
            uint32_t m = srow[x];  // position along the body, 0 head .. 255 tail
            if (a >= 255) {
                drow[x] = body_lut[m];
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
            // Smaller than they were: against a solid body a node this size
            // read as a node, against the stipple it read as a hole.
            dot(c, y0, h, px[i], py[i], rad[i] * 0.34f + 1.0f, WR_FIRE, a);
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

    float tgx = (float)w->body.target_x, tgy = (float)w->body.target_y;
    float dt = c->bg_time - c->cam_t;
    if (!c->cam_have || dt < 0.0f || dt > 1.0f) {
        c->cam_sx = tgx;
        c->cam_sy = tgy;
        c->cam_have = true;
        dt = 0.0f;
    }
    // dt of zero means the caller is not advancing bg_time; weld to the head
    // rather than stall the camera entirely.
    float k = (dt > 0.0f && c->cam_lag > 0.001f) ? 1.0f - expf(-dt / c->cam_lag) : 1.0f;
    c->cam_sx += (tgx - c->cam_sx) * k;
    c->cam_sy += (tgy - c->cam_sy) * k;
    // Leash. A long lag is what actually smooths the bob out — but on its own it
    // lets the head wander out of frame, because a good part of the head's
    // motion is the animal genuinely travelling and the camera has to go too.
    // So: follow slowly, and drag the rest of the way once the head is more than
    // WR_CAM_LEASH px off the middle. Smooth where it can be, hard where it must.
    float ox = tgx - c->cam_sx, oy = tgy - c->cam_sy;
    float lim = WR_CAM_LEASH / scale_of(c);
    float od2 = ox * ox + oy * oy;
    if (od2 > lim * lim) {
        float f = 1.0f - lim / sqrtf(od2);
        c->cam_sx += ox * f;
        c->cam_sy += oy * f;
    }
    c->cam_t = c->bg_time;
    c->cam_x = c->cam_sx;
    c->cam_y = c->cam_sy;
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

    // The background: the ball first (it moves slowly enough to be worth every
    // other frame), then this frame's haze on top of a copy of it.
    uint32_t tb0 = WR_T();
    if (c->bg_alien) {
        if (bg_noise_frame == 0xFFFFFFFFu ||
            c->frame - bg_noise_frame >= WR_BG_NOISE_EVERY) {
            build_bg_noise(c->bg_time);
            bg_noise_frame = c->frame;
        }
        memcpy(bg_field, bg_noise, WR_BG_BYTES);
    } else {
        memset(bg_field, 0, WR_BG_BYTES);
    }
    build_words(c, w);
    splat_haze(c);
    bg_row_have = -1;
    wr_us_sphere += WR_T() - tb0;

    float ring_r = (1.0f - c->flash) * 210.0f + 14.0f;
    uint32_t ring_a = (uint32_t)(c->flash * c->flash * 150.0f);

    for (int y0 = 0; y0 < WR_H; y0 += c->band_rows) {
        int h = WR_H - y0 < c->band_rows ? WR_H - y0 : c->band_rows;
        size_t n = (size_t)WR_W * h;

        c->band = c->band_mem + (size_t)c->band_parity * WR_W * c->band_rows;
        c->band_parity ^= 1;

        uint32_t tg0 = WR_T();
        band_bg(c, y0, h);
        wr_us_bg += WR_T() - tg0;

        uint32_t t0 = WR_T();
        uint32_t ga = (uint32_t)(c->globe_alpha * 256.0f);
        if (ga) {
            for (int k = 0; k < n_vis; k++) {
                if (vis_y1[k] < y0 || vis_y0[k] >= y0 + h) continue;
                int i = vis[k];
                frame_line(c, y0, h, fx[i - 1], fy[i - 1], fx[i], fy[i],
                           WR_FRAME_HALF_WIDTH, wr_frame[i].alpha * ga >> 8);
            }
        }

        uint32_t t1 = WR_T();
        band_words(c, y0, h);
        uint32_t t2 = WR_T();
        band_worm(c, y0, h, px, py, rad);
        uint32_t t3 = WR_T();
        wr_us_frame += t1 - t0; wr_us_words += t2 - t1; wr_us_worm += t3 - t2;
        if (c->anatomy && !c->xray) band_anatomy(c, y0, h, px, py, rad);
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
