// Host preview: run the sim, render frames, dump PPMs.
//
// The renderer is the same C the board runs, so what comes out here is what the
// AMOLED will show. tools/preview/run.py turns the PPMs into PNGs / a GIF.
//
//   ./preview ../../build/liam.hwrm outdir 45000 12 30

#include "wormrender.h"
#include "wormsim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Same microsecond source the firmware hands the renderer, so the per-stage
// counters mean the same thing here as they do in the board's fps line.
static uint32_t us_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)(tv.tv_sec * 1000000ull + tv.tv_usec);
}

// The board hands each band straight to the panel; here we reassemble a whole
// frame so it can be written out as an image.
static void collect_band(void *user, int y, int h, const uint16_t *pixels) {
    memcpy((uint16_t *)user + (size_t)y * WR_W, pixels,
           sizeof(uint16_t) * (size_t)WR_W * h);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <asset.hwrm> <outdir> <warmup_ticks> [n_frames] "
                "[ticks_per_frame] [view_units]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const char *outdir = argv[2];
    long warmup = atol(argv[3]);
    int n_frames = argc > 4 ? atoi(argv[4]) : 12;
    int per_frame = argc > 5 ? atoi(argv[5]) : 30;
    double view_units = argc > 6 ? atof(argv[6]) : 400.0;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)size);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) { perror("read"); return 1; }
    fclose(f);

    wm_asset a;
    if (!wm_asset_open(&a, buf, (size_t)size)) { fprintf(stderr, "bad asset\n"); return 1; }

    wm_world *w = malloc(sizeof(wm_world));
    wm_world_init(w, &a, a.seed, malloc(wm_world_bytes(&a)));

    // Warm up past the initial tangle: the IK chain starts as random jitter and
    // needs a moment to straighten into an animal.
    for (long t = 0; t < warmup; t++) wm_world_tick(w);

    uint16_t *fb = malloc(sizeof(uint16_t) * WR_W * WR_H);
    const int band_rows = 32;
    wr_clock = us_now;
    wr_build_globe();
    wr_ctx ctx;
    wr_init(&ctx, malloc(wr_scratch_bytes(band_rows)), band_rows,
            malloc(wr_bg_bytes()));
    ctx.view_units = view_units;
    ctx.title = "Liam";
    ctx.subtitle = "flask_1  gen-0007";
    ctx.title_alpha = 1.0f;
    float hold = 2.5f;
    ctx.invert = getenv("WR_INVERT") != NULL;   // preview the touch states
    ctx.xray = getenv("WR_XRAY") != NULL;
    if (getenv("WR_SOLID")) ctx.stipple = 0.0f;      // the body without the screen
    if (getenv("WR_NOBG")) ctx.bg_alien = false;     // and without the ball
    if (getenv("WR_NOHAZE")) ctx.haze = 0.0f;
    if (getenv("WR_NOANAT")) ctx.anatomy = false;

    uint32_t t_draw = 0;
    for (int i = 0; i < n_frames; i++) {
        for (int t = 0; t < per_frame; t++) {
            wm_world_tick(w);
            wm_eaten got[WM_EATEN_CAP];
            int n = wm_world_drain_eaten(w, got, WM_EATEN_CAP);
            for (int k = 0; k < n; k++) {
                uint32_t len;
                const char *s = wm_str(&a.tok_text, got[k].tok, &len);
                fprintf(stderr, "  frame %d: ate %.*s\n", i, (int)len, s);
                ctx.flash = 1.0f;
            }
        }
        uint32_t td = us_now();
        wr_draw_banded(&ctx, w, collect_band, fb);
        t_draw += us_now() - td;
        if (getenv("WR_CAM")) {
            // How fast does the camera actually move? Words drift up at a fixed
            // 15 world units a second, but the camera rides the head, so what a
            // word does on screen is that minus whatever the worm is doing.
            static float pcx, pcy; static int have;
            float s = (float)WR_W / ctx.view_units;
            float ox = ((float)w->body.target_x - ctx.cam_x) * s;
            float oy = ((float)w->body.target_y - ctx.cam_y) * s;
            if (have)
                printf("%d off %.1f %.1f cam %.1f %.1f  d/s %.1f %.1f px\n", i, ox, oy,
                       ctx.cam_x, ctx.cam_y,
                       (ctx.cam_x - pcx) * s * (float)WM_BODY_TICK_HZ / per_frame,
                       (ctx.cam_y - pcy) * s * (float)WM_BODY_TICK_HZ / per_frame);
            pcx = ctx.cam_x; pcy = ctx.cam_y; have = 1;
        }
        // Same law as firmware/main/main.c, on this frame's simulated dt.
        float dt = (float)per_frame / (float)WM_BODY_TICK_HZ;
        ctx.bg_time += dt;
        ctx.flash -= ctx.flash * dt * 3.2f;
        if (ctx.flash < 0.002f) ctx.flash = 0.0f;
        // Same hold-then-fade as firmware/main/main.c.
        hold -= dt;
        if (hold < 0.0f) ctx.title_alpha -= dt * 0.7f;
        if (ctx.title_alpha < 0.0f) ctx.title_alpha = 0.0f;

        // "-" for the outdir renders without writing anything: the point is the
        // clock, and a 650 KB PPM per frame swamps it.
        if (outdir[0] == '-' && !outdir[1]) continue;

        char name[512];
        snprintf(name, sizeof(name), "%s/frame_%04d.ppm", outdir, i);
        FILE *o = fopen(name, "wb");
        if (!o) { perror(name); return 1; }
        fprintf(o, "P6\n%d %d\n255\n", WR_W, WR_H);
        for (int p = 0; p < WR_W * WR_H; p++) {
            uint16_t v = fb[p];
            unsigned char rgb[3] = {
                (unsigned char)(((v >> 11) & 0x1F) * 255 / 31),
                (unsigned char)(((v >> 5) & 0x3F) * 255 / 63),
                (unsigned char)((v & 0x1F) * 255 / 31),
            };
            fwrite(rgb, 1, 3, o);
        }
        fclose(o);
    }
    fprintf(stderr, "wrote %d frames to %s (tick %lld, view %.0f world units)\n",
            n_frames, outdir, (long long)w->tick_count, view_units);
    fprintf(stderr, "draw %.3f ms/frame\n", t_draw / 1000.0 / n_frames);
    fprintf(stderr, "per frame: ball %.2f  bg %.2f  frame %.2f  words %.2f  worm %.2f ms\n",
            wr_us_sphere / 1000.0 / n_frames,
            wr_us_bg / 1000.0 / n_frames, wr_us_frame / 1000.0 / n_frames,
            wr_us_words / 1000.0 / n_frames, wr_us_worm / 1000.0 / n_frames);
    return 0;
}
