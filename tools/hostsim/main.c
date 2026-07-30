// Host driver for the C sim. Emits a trace that tools/crosscheck.py diffs
// against the Python original, so the port is verified before it ever touches
// hardware.
//
//   ./hostsim ../../build/liam.hwrm 20000
//
// Body state prints as %a (hex float) — the whole point is to catch a
// discrepancy in the last bit, which %f would hide.

#include "wormsim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <asset.hwrm> <n_ticks> [state_every]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    long n_ticks = atol(argv[2]);
    long state_every = argc > 3 ? atol(argv[3]) : 600;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)size);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) { perror("read"); return 1; }
    fclose(f);

    wm_asset a;
    if (!wm_asset_open(&a, buf, (size_t)size)) {
        fprintf(stderr, "bad asset\n");
        return 1;
    }
    fprintf(stderr, "asset: %u vocab, %u neurons, %u edges, %u sentences, %u tokens, seed %u\n",
            a.n_vocab, a.n_neurons, a.n_edges, a.n_sentences, a.n_tokens, a.seed);

    // Probe mode: dump the 12-dim chemosensory vector for a handful of
    // (word, history) pairs so tools/embedprobe.py can diff them against
    // embed_batch() bit for bit. Isolates the embedding forward pass from the
    // rest of the sim when hunting a divergence.
    if (argc > 4 && strcmp(argv[4], "embed") == 0) {
        uint32_t probes[][6] = {
            {100, 0, 0, 0, 0, 0}, {101, 100, 0, 0, 0, 0},
            {205, 100, 101, 0, 0, 0}, {311, 205, 100, 101, 102, 0},
            {412, 311, 205, 100, 101, 102}, {77, 412, 311, 205, 100, 101},
        };
        int n_hist[] = {0, 1, 2, 4, 5, 5};
        for (int p = 0; p < 6; p++) {
            double pca[WM_N_PC];
            if (!wm_chemo_embed(&a, probes[p][0], &probes[p][1], n_hist[p], pca)) {
                printf("PROBE %u OOV\n", probes[p][0]);
                continue;
            }
            printf("PROBE %u %d", probes[p][0], n_hist[p]);
            for (int i = 0; i < WM_N_PC; i++) printf(" %a", pca[i]);
            printf("\n");
        }
        return 0;
    }

    wm_world *w = malloc(sizeof(wm_world));
    void *storage = malloc(wm_world_bytes(&a));
    wm_world_init(w, &a, a.seed, storage);

    int max_lines = 0, max_words = 0;
    for (long t = 0; t < n_ticks; t++) {
        wm_world_tick(w);
        // The scroller has fixed line slots; if either of these ever reaches
        // its cap a sentence gets skipped or truncated silently.
        if (w->scroller.n_lines > max_lines) max_lines = w->scroller.n_lines;
        for (int li = 0; li < w->scroller.n_lines; li++)
            if (w->scroller.lines[li].n_words > max_words)
                max_words = w->scroller.lines[li].n_words;

        if (wm_scroller_exhausted(&w->scroller)) {
            fprintf(stderr, "corpus exhausted at tick %lld\n", (long long)w->tick_count);
            break;
        }

        wm_eaten got[WM_EATEN_CAP];
        int n = wm_world_drain_eaten(w, got, WM_EATEN_CAP);
        for (int i = 0; i < n; i++) {
            uint32_t len;
            const char *s = wm_str(&a.tok_text, got[i].tok, &len);
            printf("EAT %lld %d %d %.*s\n", (long long)w->tick_count,
                   got[i].line_id, got[i].word_idx, (int)len, s);
        }

        if (state_every > 0 && (t % state_every) == 0) {
            printf("STATE %lld %a %a %a %a\n", (long long)w->tick_count,
                   w->body.target_x, w->body.target_y, w->body.facing_dir, w->body.speed);
        }
    }

    fprintf(stderr, "peak %d/%d active lines, %d/%d words per line\n",
            max_lines, WM_MAX_ACTIVE_LINES, max_words, WM_MAX_LINE_WORDS);
    printf("END %lld\n", (long long)w->tick_count);
    return 0;
}
