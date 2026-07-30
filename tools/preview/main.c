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
    wr_ctx ctx;
    wr_init(&ctx, malloc(wr_scratch_bytes(band_rows)), band_rows);
    ctx.view_units = view_units;

    for (int i = 0; i < n_frames; i++) {
        for (int t = 0; t < per_frame; t++) {
            wm_world_tick(w);
            wm_eaten got[WM_EATEN_CAP];
            int n = wm_world_drain_eaten(w, got, WM_EATEN_CAP);
            for (int k = 0; k < n; k++) {
                uint32_t len;
                const char *s = wm_str(&a.tok_text, got[k].tok, &len);
                fprintf(stderr, "  frame %d: ate %.*s\n", i, (int)len, s);
            }
        }
        wr_draw_banded(&ctx, w, collect_band, fb);

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
    return 0;
}
