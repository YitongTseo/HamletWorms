// wormsim — a portable C port of HamletRNAWorld v7's simulation core.
//
// No ESP-IDF dependencies: this builds for the host (see tools/hostsim) so the
// trajectory can be diffed against the Python original before it ever reaches
// the board.
//
// Everything is `double`, deliberately. Python/numpy is float64 throughout, and
// the connectome's fire test is a hard threshold (psyn > 30) — running the body
// in float32 would flip a neuron within seconds and the worm would walk away
// from its server-side twin. The ESP32-S3 has no double FPU so these are
// software-emulated, but the workload is tiny: ~240k double-ops/sec for the
// 200-segment chain at 60 Hz, against a 240 MHz core.
//
// IEEE 754 add/mul/sqrt are exactly specified, so those already match bit for
// bit. What does NOT match across libm implementations is sin/cos/atan2/hypot/
// exp — see wm_math.h, where they are isolated behind a swappable layer.

#ifndef WORMSIM_H
#define WORMSIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// --- world constants (sim/world.py, sim/text_scroller.py) --------------------
#define WM_WORLD_W 1600.0
#define WM_WORLD_H 1000.0
#define WM_BODY_TICK_HZ 60
#define WM_BRAIN_TICK_PERIOD 30
#define WM_BODY_DT (1.0 / 60.0)
#define WM_FOOD_SENSE_RADIUS 200.0
#define WM_FOOD_EAT_RADIUS 20.0
#define WM_STIM_LINGER_TICKS 120  // int(2.0 * 60)

#define WM_SCROLL_SPEED 15.0
#define WM_SPAWN_Y (WM_WORLD_H - 20.0)
#define WM_KILL_Y (-80.0)
#define WM_SPAWN_INTERVAL 4.5
#define WM_CHAR_W 11.0
#define WM_WORD_GAP 70.0
#define WM_CENTER_X (WM_WORLD_W / 2.0)

#define WM_N_SEGMENTS 200
#define WM_SEGMENT_SIZE 4.0
#define WM_FIRE_THRESHOLD 30.0

// --- embedding dimensions (server/embedding.py) -----------------------------
#define WM_D_EMB 11
#define WM_HISTORY 5
#define WM_D_HIST_CONCAT (WM_D_EMB * WM_HISTORY)  // 55
#define WM_D_FUSE (WM_D_EMB * 2)                  // 22
#define WM_N_TAGS 12
#define WM_D_POS_IN (WM_N_TAGS * (WM_HISTORY + 1))  // 72
#define WM_D_POS_HID 16
#define WM_N_PC 12  // 11 embedding dims + 1 POS dim

#define WM_OOV 0xFFFF

// ---------------------------------------------------------------------------
// Asset file (tools/bake.py output)
// ---------------------------------------------------------------------------
typedef struct {
    const char *blob;
    const uint32_t *off;
    uint32_t count;
} wm_strtab;

typedef struct {
    const void *base;
    size_t size;

    const char *meta;  // NUL-terminated JSON

    wm_strtab vocab;
    uint32_t n_vocab;
    const double *etable;  // [n_vocab][WM_D_EMB]

    // shared nets — the part of the embedder that could not be precomputed
    const double *W_Hh, *b_Hh, *W_Hf, *b_Hf, *W_P1, *b_P1, *W_P2, *b_P2;

    wm_strtab neurons;
    uint32_t n_neurons;
    const uint8_t *is_muscle;

    const uint32_t *row_start;  // [n_neurons + 1]
    const uint16_t *syn_col;
    const double *syn_w;
    uint32_t n_edges;

    uint32_t n_presyn;
    const uint16_t *presyn_idx;  // weights.json key order, for rand_excite

    uint32_t n_muscle_visits;
    const uint16_t *muscle_idx;
    const uint8_t *muscle_side;  // 0 = left, 1 = right, 2 = neither

    uint16_t chemo_pair[WM_N_PC][2];

    uint32_t n_hunger, n_nose, n_food;
    const uint16_t *hunger_idx, *nose_idx, *food_idx;

    uint32_t n_sentences, n_tokens;
    const uint32_t *sent_start;  // [n_sentences + 1]
    const uint16_t *tok_vocab;   // WM_OOV when the word has no nomic vector
    const uint8_t *tok_pos;
    const uint8_t *tok_len;  // CODE POINTS, not bytes — the em dash is 3 bytes
    const uint8_t *sent_edible;
    wm_strtab tok_text;  // display form, original capitalisation

    uint32_t seed;
    uint32_t mvulva_idx;  // UINT32_MAX when absent
} wm_asset;

// Parses in place; `data` must outlive the asset (flash-mapped is fine).
bool wm_asset_open(wm_asset *a, const void *data, size_t size);
const char *wm_str(const wm_strtab *t, uint32_t i, uint32_t *len_out);

// ---------------------------------------------------------------------------
// MT19937 — a faithful port of CPython's random.Random, seeding included.
// The worm draws 800 uniform(-1,1) for its initial body pose and then 40
// choice() calls for rand_excite, in that order. Get this wrong and every
// trajectory is wrong from tick zero.
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t mt[624];
    int index;
} wm_rng;

void wm_rng_seed(wm_rng *r, uint32_t seed);
uint32_t wm_rng_u32(wm_rng *r);
double wm_rng_double(wm_rng *r);            // CPython random.random()
double wm_rng_uniform(wm_rng *r, double a, double b);
uint32_t wm_rng_below(wm_rng *r, uint32_t n);  // CPython _randbelow

// ---------------------------------------------------------------------------
// Connectome
// ---------------------------------------------------------------------------
typedef struct {
    const wm_asset *a;
    double *psyn;  // [n_neurons][2], interleaved
    int this_state, next_state;
    double accum_left, accum_right;
} wm_brain;

void wm_brain_init(wm_brain *b, const wm_asset *a, double *psyn_storage);
void wm_brain_accumulate(wm_brain *b, uint32_t pre, double scale);
void wm_brain_run(wm_brain *b);
void wm_brain_rand_excite(wm_brain *b, wm_rng *r, int k);

// ---------------------------------------------------------------------------
// Body — worm-sim's trailing IK chain
// ---------------------------------------------------------------------------
typedef struct {
    double hx[WM_N_SEGMENTS], hy[WM_N_SEGMENTS];
    double tx[WM_N_SEGMENTS], ty[WM_N_SEGMENTS];
    double facing_dir, target_dir, speed, target_speed, speed_change;
    double target_x, target_y;
} wm_body;

void wm_body_init(wm_body *w, double ox, double oy, wm_rng *r);
void wm_body_consume_motor(wm_body *w, double accum_left, double accum_right);
void wm_body_step(wm_body *w);

// ---------------------------------------------------------------------------
// Text scroller
// ---------------------------------------------------------------------------
#define WM_MAX_ACTIVE_LINES 64
#define WM_MAX_LINE_WORDS 96

typedef struct {
    uint32_t tok;  // index into asset.tok_* arrays
    double x, y;
    bool eaten;
} wm_word;

typedef struct {
    int32_t line_id;
    uint32_t sent_idx;
    bool edible;
    uint16_t n_words;
    wm_word w[WM_MAX_LINE_WORDS];
} wm_line;

typedef struct {
    const wm_asset *a;
    wm_line lines[WM_MAX_ACTIVE_LINES];
    int n_lines;
    uint32_t sent_idx;
    int32_t line_id;
    double elapsed, next_spawn;
    bool loop;
} wm_scroller;

void wm_scroller_init(wm_scroller *s, const wm_asset *a, bool loop);
void wm_scroller_step(wm_scroller *s, double dt);
bool wm_scroller_exhausted(const wm_scroller *s);

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t tok;      // asset token index of the eaten word
    int32_t line_id;
    int32_t word_idx;
} wm_eaten;

#define WM_EATEN_CAP 64

typedef struct {
    const wm_asset *a;
    wm_rng rng;
    wm_brain brain;
    wm_body body;
    wm_scroller scroller;

    bool stim_hunger, stim_nose_touch, stim_food_sense;
    int64_t stim_linger_until;
    int64_t tick_count;
    int64_t brain_tick_count;

    // Most-recent-first, exactly like World._recent_eaten. This IS the worm's
    // memory in v7 — there is no decaying residual any more.
    uint32_t recent[WM_HISTORY];
    int n_recent;

    wm_eaten eaten[WM_EATEN_CAP];
    int n_eaten;

    double *psyn_storage;

    // _chemo_pulse builds a plain dict and stimulate_weighted then iterates it.
    // Python dicts iterate in INSERTION order, and insertion happens only when
    // a neuron's contribution is > 0, so the order is data-dependent. These
    // three arrays reproduce that exactly: value, insertion sequence, membership.
    double *chemo_val;
    uint16_t *chemo_order;
    uint8_t *chemo_seen;
    int n_chemo;
} wm_world;

// `storage` must hold wm_world_bytes(a) bytes and outlive the world.
size_t wm_world_bytes(const wm_asset *a);
void wm_world_init(wm_world *w, const wm_asset *a, uint32_t seed, void *storage);
void wm_world_tick(wm_world *w);
int wm_world_drain_eaten(wm_world *w, wm_eaten *out, int cap);

// ---------------------------------------------------------------------------
// Chemosensation — the embedding forward pass, minus the 512->11 projection
// that tools/bake.py already folded into asset.etable.
// ---------------------------------------------------------------------------
// Fills `out[WM_N_PC]` for one on-screen word given the worm's eaten history.
// Returns false when the word is OOV, matching embed() returning None.
bool wm_chemo_embed(const wm_asset *a, uint32_t tok,
                    const uint32_t *recent, int n_recent, double *out);

#endif  // WORMSIM_H
