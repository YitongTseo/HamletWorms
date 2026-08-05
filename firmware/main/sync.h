#ifndef SYNC_H
#define SYNC_H

#include <stdint.h>

#include "wormsim.h"

// Called when a newer generation has been downloaded and parsed. `weights` is
// n_edges doubles in CSR order and stays valid; the callback is on the sync
// task, so it should hand off rather than block.
typedef void (*sync_genome_fn)(const double *weights, uint32_t epoch, void *ctx);

typedef struct {
    char ssid[33];
    char pass[65];
    const char *flask;   // e.g. "flask_1"
    const char *worm;    // e.g. "Liam"
    uint32_t epoch;      // generation this board is currently running
    int poll_minutes;
    const wm_asset *asset;
    sync_genome_fn on_genome;
    void *ctx;
} sync_cfg_t;

// Best-effort. Returns immediately; everything happens on its own task, and a
// board with no network keeps running the genome baked into its flash.
void sync_start(const sync_cfg_t *cfg);

#endif
