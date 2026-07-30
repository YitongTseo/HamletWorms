// sim/connectome.py — integrate-and-fire, double-buffered.
//
// The one thing to be careful about here is *order*. Float addition is not
// associative, so synapses must land in the order CPython's dict iteration
// produced (bake.py preserved it in the CSR rows) and neurons must be visited
// in sorted() order (bake.py sorted them).

#include "wm_math.h"
#include "wormsim.h"

#include <string.h>

#define PS(b, n, s) ((b)->psyn[(size_t)(n) * 2 + (s)])

void wm_brain_init(wm_brain *b, const wm_asset *a, double *psyn_storage) {
    b->a = a;
    b->psyn = psyn_storage;
    memset(b->psyn, 0, sizeof(double) * 2 * a->n_neurons);
    b->this_state = 0;
    b->next_state = 1;
    b->accum_left = 0.0;
    b->accum_right = 0.0;
}

void wm_brain_accumulate(wm_brain *b, uint32_t pre, double scale) {
    const wm_asset *a = b->a;
    if (pre >= a->n_neurons) return;
    uint32_t s = a->row_start[pre], e = a->row_start[pre + 1];
    int ns = b->next_state;
    for (uint32_t i = s; i < e; i++)
        PS(b, a->syn_col[i], ns) += a->syn_w[i] * scale;
}

static void fire_neuron(wm_brain *b, uint32_t n) {
    // Guarding MVULVA is dead code in practice — run() already skips anything
    // whose name starts with a muscle prefix, and MVULVA starts with "MVU".
    // Kept because the Python keeps it, and cheap enough not to care.
    if (n == b->a->mvulva_idx) return;
    wm_brain_accumulate(b, n, 1.0);
    PS(b, n, b->next_state) = 0.0;
}

static void motorcontrol(wm_brain *b) {
    const wm_asset *a = b->a;
    b->accum_left = 0.0;
    b->accum_right = 0.0;
    int ns = b->next_state;
    // MUSCLE_LIST visits MDL21 and MVL21 twice (M_RIGHT carries the left-side
    // names — a typo inherited from the GoPiGo original that worm-sim and v7
    // both preserved). The second visit reads the value this loop already
    // zeroed and adds 0.0 to accum_LEFT, because `m in _left_set` matches
    // first. MDR21/MVR21 are consequently never visited at all: their charge
    // accumulates forever and is never read. Both quirks are load-bearing.
    for (uint32_t i = 0; i < a->n_muscle_visits; i++) {
        uint32_t m = a->muscle_idx[i];
        double v = PS(b, m, ns);
        if (a->muscle_side[i] == 0) b->accum_left += v;
        else if (a->muscle_side[i] == 1) b->accum_right += v;
        PS(b, m, ns) = 0.0;
    }
}

void wm_brain_run(wm_brain *b) {
    const wm_asset *a = b->a;
    int ts = b->this_state;
    for (uint32_t n = 0; n < a->n_neurons; n++) {
        if (a->is_muscle[n]) continue;
        if (PS(b, n, ts) > WM_FIRE_THRESHOLD) fire_neuron(b, n);
    }
    motorcontrol(b);
    int ns = b->next_state;
    for (uint32_t n = 0; n < a->n_neurons; n++) PS(b, n, ts) = PS(b, n, ns);
    b->this_state = ns;
    b->next_state = ts;
}

void wm_brain_rand_excite(wm_brain *b, wm_rng *r, int k) {
    // choice() over list(weights.keys()) — 300 entries, not the 396 neurons.
    // The length sets the rejection-sampling bit width, so using the wrong list
    // consumes a different number of draws and desyncs the whole RNG stream.
    const wm_asset *a = b->a;
    for (int i = 0; i < k; i++)
        wm_brain_accumulate(b, a->presyn_idx[wm_rng_below(r, a->n_presyn)], 1.0);
}
