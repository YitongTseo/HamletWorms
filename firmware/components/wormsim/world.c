// sim/world.py — brain slow, body fast, food in between.

#include "wm_math.h"
#include "wormsim.h"

#include <string.h>

size_t wm_world_bytes(const wm_asset *a) {
    size_t n = a->n_neurons;
    return sizeof(double) * 2 * n     // psyn, double-buffered
         + sizeof(double) * n         // chemo values
         + sizeof(uint16_t) * n       // chemo insertion order
         + sizeof(uint8_t) * n;       // chemo membership
}

void wm_world_init(wm_world *w, const wm_asset *a, uint32_t seed, void *storage) {
    memset(w, 0, sizeof(*w));
    w->a = a;

    uint8_t *p = (uint8_t *)storage;
    size_t n = a->n_neurons;
    w->psyn_storage = (double *)p;   p += sizeof(double) * 2 * n;
    w->chemo_val = (double *)p;      p += sizeof(double) * n;
    w->chemo_order = (uint16_t *)p;  p += sizeof(uint16_t) * n;
    w->chemo_seen = (uint8_t *)p;

    // Draw order matters and follows World.__post_init__, not field order:
    // the brain is constructed first but draws nothing, then the body takes
    // 800 uniforms, and only then does rand_excite take its 40 choices.
    wm_rng_seed(&w->rng, seed);
    wm_brain_init(&w->brain, a, w->psyn_storage);
    wm_body_init(&w->body, WM_WORLD_W / 2.0, WM_WORLD_H / 2.0, &w->rng);
    wm_brain_rand_excite(&w->brain, &w->rng, 40);

    // loop=False is generation semantics: one full pass of the play, which is
    // what the server runs when WORMLET_GENERATIONS_ENABLED=1 and what produced
    // the champion we baked.
    wm_scroller_init(&w->scroller, a, false);

    w->stim_hunger = true;
}

static void stimulate(wm_world *w, const uint16_t *idx, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) wm_brain_accumulate(&w->brain, idx[i], 1.0);
    wm_brain_run(&w->brain);
}

static void chemo_add(wm_world *w, uint16_t neuron, double v) {
    // Emulates `out[n] = out.get(n, 0.0) + v` on a Python dict: first insertion
    // fixes the iteration position, and insertion only happens when v > 0.
    if (!w->chemo_seen[neuron]) {
        w->chemo_seen[neuron] = 1;
        w->chemo_val[neuron] = 0.0;
        w->chemo_order[w->n_chemo++] = neuron;
    }
    w->chemo_val[neuron] += v;
}

// _compute_smells fused with _chemo_pulse. Splitting them the way the Python
// does would need a materialised smell list; the arithmetic and the ordering
// are identical either way because sensed_smells is keyed uniquely per word and
// therefore iterates in in-range order.
static void compute_smells(wm_world *w) {
    const wm_asset *a = w->a;
    double wx = w->body.target_x, wy = w->body.target_y;

    w->n_chemo = 0;
    memset(w->chemo_seen, 0, a->n_neurons);

    for (int li = 0; li < w->scroller.n_lines; li++) {
        const wm_line *L = &w->scroller.lines[li];
        if (!L->edible) continue;  // set-dressing: no eating, no smelling
        for (uint16_t wi = 0; wi < L->n_words; wi++) {
            const wm_word *word = &L->w[wi];
            if (word->eaten) continue;

            double dx = word->x - wx;
            double dy = word->y - wy;
            double d = wm_hypot(dx, dy);
            if (d > WM_FOOD_SENSE_RADIUS) continue;

            double pca[WM_N_PC];
            if (!wm_chemo_embed(a, word->tok, w->recent, w->n_recent, pca)) continue;

            double distance_factor = 1.0 - (d / WM_FOOD_SENSE_RADIUS);
            double food_angle = wm_atan2(dy, dx);
            double angle_diff =
                wm_pymod(food_angle - w->body.facing_dir + WM_PI, 2.0 * WM_PI) - WM_PI;
            double direction_factor = 0.5 + 0.5 * wm_sin(angle_diff);

            // compute_pca_activation: v = pca[i] * intensity, THEN split L/R.
            double dirL = direction_factor, dirR = 1.0 - direction_factor;
            for (int i = 0; i < WM_N_PC; i++) {
                double v = pca[i] * distance_factor;
                double vl = v * dirL, vr = v * dirR;
                if (vl > 0.0) chemo_add(w, a->chemo_pair[i][0], vl);
                if (vr > 0.0) chemo_add(w, a->chemo_pair[i][1], vr);
            }
        }
    }

    for (int i = 0; i < w->n_chemo; i++) {
        uint16_t n = w->chemo_order[i];
        if (w->chemo_val[n] > 1.0) w->chemo_val[n] = 1.0;  // saturate per neuron
    }
}

static void check_food(wm_world *w) {
    double wx = w->body.target_x, wy = w->body.target_y;
    for (int li = 0; li < w->scroller.n_lines; li++) {
        wm_line *L = &w->scroller.lines[li];
        if (!L->edible) continue;
        for (uint16_t wi = 0; wi < L->n_words; wi++) {
            wm_word *word = &L->w[wi];
            if (word->eaten) continue;
            double d = wm_hypot(wx - word->x, wy - word->y);
            if (d > WM_FOOD_SENSE_RADIUS) continue;

            w->stim_food_sense = true;
            w->stim_linger_until = w->tick_count + WM_STIM_LINGER_TICKS;
            if (d > WM_FOOD_EAT_RADIUS) continue;

            word->eaten = true;
            if (w->n_eaten < WM_EATEN_CAP) {
                w->eaten[w->n_eaten].tok = word->tok;
                w->eaten[w->n_eaten].line_id = L->line_id;
                w->eaten[w->n_eaten].word_idx = wi;
                w->n_eaten++;
            }
            // _recent_eaten.insert(0, word) then truncate to HISTORY. This IS
            // the worm's memory in v7 — it feeds the embedding, so the same
            // word tastes different depending on what was eaten before it.
            for (int k = (w->n_recent < WM_HISTORY ? w->n_recent : WM_HISTORY - 1); k > 0; k--)
                w->recent[k] = w->recent[k - 1];
            w->recent[0] = word->tok;
            if (w->n_recent < WM_HISTORY) w->n_recent++;
        }
    }
}

static void check_walls(wm_world *w) {
    bool hit = false;
    if (w->body.target_x < 0.0) { w->body.target_x = 0.0; hit = true; }
    else if (w->body.target_x > WM_WORLD_W) { w->body.target_x = WM_WORLD_W; hit = true; }
    if (w->body.target_y < 0.0) { w->body.target_y = 0.0; hit = true; }
    else if (w->body.target_y > WM_WORLD_H) { w->body.target_y = WM_WORLD_H; hit = true; }
    if (hit) {
        w->stim_nose_touch = true;
        w->stim_linger_until = w->tick_count + WM_STIM_LINGER_TICKS;
    }
}

void wm_world_tick(wm_world *w) {
    const wm_asset *a = w->a;

    wm_scroller_step(&w->scroller, WM_BODY_DT);

    if (w->tick_count % WM_BRAIN_TICK_PERIOD == 0) {
        // Smells are computed only on brain ticks: the brain is the only reader
        // and the embedding pass is the dominant per-tick cost. Note this reads
        // the body position from BEFORE this tick's step().
        compute_smells(w);

        if (w->stim_hunger) stimulate(w, a->hunger_idx, a->n_hunger);
        if (w->stim_nose_touch) stimulate(w, a->nose_idx, a->n_nose);
        if (w->stim_food_sense) stimulate(w, a->food_idx, a->n_food);

        if (w->n_chemo > 0) {
            for (int i = 0; i < w->n_chemo; i++) {
                uint16_t n = w->chemo_order[i];
                double v = w->chemo_val[n];
                if (v > 0.0) wm_brain_accumulate(&w->brain, n, v);
            }
            wm_brain_run(&w->brain);
        }

        w->brain_tick_count++;
        wm_body_consume_motor(&w->body, w->brain.accum_left, w->brain.accum_right);
    }

    wm_body_step(&w->body);
    check_food(w);
    check_walls(w);

    if (w->tick_count >= w->stim_linger_until) {
        w->stim_hunger = true;
        w->stim_nose_touch = false;
        w->stim_food_sense = false;
    }

    w->tick_count++;
}

int wm_world_drain_eaten(wm_world *w, wm_eaten *out, int cap) {
    int n = w->n_eaten < cap ? w->n_eaten : cap;
    memcpy(out, w->eaten, sizeof(wm_eaten) * (size_t)n);
    w->n_eaten = 0;
    return n;
}

void wm_world_poke(wm_world *w) {
    // Exactly what _check_walls does on contact: raise the flag and hold it for
    // the linger window, so the next brain tick fires the mechanosensors.
    w->stim_nose_touch = true;
    w->stim_linger_until = w->tick_count + WM_STIM_LINGER_TICKS;
}
