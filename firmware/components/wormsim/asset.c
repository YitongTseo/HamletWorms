// Parser for the .hwrm asset produced by tools/bake.py.
//
// Zero-copy: every field points into the caller's buffer, which on the board is
// a memory-mapped flash partition. Nothing is allocated and nothing is byte-
// swapped — the file is little-endian and so is the ESP32-S3.

#include "wormsim.h"

#include <string.h>

#define MAGIC 0x4D525748u  // "HWRM" little-endian

// Cursor over one section. All alignment in bake.py is relative to the start of
// the section, and every section starts 16-byte aligned, so tracking the
// section-relative offset is enough.
typedef struct {
    const uint8_t *base;
    uint32_t off, size;
    bool ok;
} cur;

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static const void *take(cur *c, uint32_t nbytes) {
    if (!c->ok || c->off + nbytes > c->size) {
        c->ok = false;
        return NULL;
    }
    const void *p = c->base + c->off;
    c->off += nbytes;
    return p;
}

static uint32_t take_u32(cur *c) {
    const void *p = take(c, 4);
    return p ? rd_u32((const uint8_t *)p) : 0;
}

static void align_to(cur *c, uint32_t n) {
    while (c->off % n) c->off++;
}

static void take_strtab(cur *c, wm_strtab *t) {
    uint32_t count = take_u32(c);
    uint32_t blob_len = take_u32(c);
    t->count = count;
    t->off = (const uint32_t *)take(c, 4 * (count + 1));
    t->blob = (const char *)take(c, blob_len);
    align_to(c, 4);
    if (!t->off || !t->blob) c->ok = false;
}

const char *wm_str(const wm_strtab *t, uint32_t i, uint32_t *len_out) {
    if (i >= t->count) {
        if (len_out) *len_out = 0;
        return "";
    }
    if (len_out) *len_out = t->off[i + 1] - t->off[i];
    return t->blob + t->off[i];
}

static bool find_section(const uint8_t *d, size_t size, uint32_t n_sections,
                         const char *tag, cur *out) {
    char want[8];
    size_t tl = strlen(tag);
    for (size_t i = 0; i < 8; i++) want[i] = i < tl ? tag[i] : ' ';

    const uint8_t *toc = d + 16;
    for (uint32_t i = 0; i < n_sections; i++) {
        const uint8_t *e = toc + 16 * i;
        if (memcmp(e, want, 8) != 0) continue;
        uint32_t off = rd_u32(e + 8), len = rd_u32(e + 12);
        if ((size_t)off + len > size) return false;
        out->base = d + off;
        out->off = 0;
        out->size = len;
        out->ok = true;
        return true;
    }
    return false;
}

bool wm_asset_open(wm_asset *a, const void *data, size_t size) {
    memset(a, 0, sizeof(*a));
    const uint8_t *d = (const uint8_t *)data;
    if (size < 16 || rd_u32(d) != MAGIC) return false;
    if (rd_u32(d + 4) != 1) return false;
    uint32_t n_sections = rd_u32(d + 8);
    if (size < 16u + 16u * n_sections) return false;

    a->base = data;
    a->size = size;
    a->mvulva_idx = UINT32_MAX;

    cur c;

    if (find_section(d, size, n_sections, "META", &c)) {
        wm_strtab t;
        take_strtab(&c, &t);
        a->meta = t.count ? t.blob : "";
    }

    if (!find_section(d, size, n_sections, "VOCAB", &c)) return false;
    take_strtab(&c, &a->vocab);
    a->n_vocab = a->vocab.count;
    if (!c.ok) return false;

    if (!find_section(d, size, n_sections, "ETABLE", &c)) return false;
    {
        uint32_t n = take_u32(&c), dim = take_u32(&c);
        if (n != a->n_vocab || dim != WM_D_EMB) return false;
        a->etable = (const double *)take(&c, (uint32_t)(8u * n * dim));
        if (!c.ok) return false;
    }

    if (!find_section(d, size, n_sections, "NET", &c)) return false;
    {
        // Same order bake.py writes them in.
        struct { const double **dst; uint32_t n; } fields[] = {
            {&a->W_Hh, WM_D_HIST_CONCAT * WM_D_EMB}, {&a->b_Hh, WM_D_EMB},
            {&a->W_Hf, WM_D_FUSE * WM_D_EMB},        {&a->b_Hf, WM_D_EMB},
            {&a->W_P1, WM_D_POS_IN * WM_D_POS_HID},  {&a->b_P1, WM_D_POS_HID},
            {&a->W_P2, WM_D_POS_HID * 1},            {&a->b_P2, 1},
        };
        for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
            *fields[i].dst = (const double *)take(&c, 8 * fields[i].n);
        if (!c.ok) return false;
    }

    if (!find_section(d, size, n_sections, "NEURONS", &c)) return false;
    take_strtab(&c, &a->neurons);
    a->n_neurons = a->neurons.count;
    a->is_muscle = (const uint8_t *)take(&c, a->n_neurons);
    if (!c.ok) return false;
    for (uint32_t i = 0; i < a->n_neurons; i++) {
        uint32_t len;
        const char *nm = wm_str(&a->neurons, i, &len);
        if (len == 6 && memcmp(nm, "MVULVA", 6) == 0) {
            a->mvulva_idx = i;
            break;
        }
    }

    if (!find_section(d, size, n_sections, "SYNAPSE", &c)) return false;
    {
        uint32_t nn = take_u32(&c);
        a->n_edges = take_u32(&c);
        if (nn != a->n_neurons) return false;
        a->row_start = (const uint32_t *)take(&c, 4 * (nn + 1));
        a->syn_col = (const uint16_t *)take(&c, 2 * a->n_edges);
        align_to(&c, 8);
        a->syn_w = (const double *)take(&c, 8 * a->n_edges);
        if (!c.ok) return false;
    }

    if (!find_section(d, size, n_sections, "MUSCLES", &c)) return false;
    a->n_muscle_visits = take_u32(&c);
    a->muscle_idx = (const uint16_t *)take(&c, 2 * a->n_muscle_visits);
    a->muscle_side = (const uint8_t *)take(&c, a->n_muscle_visits);
    if (!c.ok) return false;

    if (!find_section(d, size, n_sections, "PRESYN", &c)) return false;
    a->n_presyn = take_u32(&c);
    a->presyn_idx = (const uint16_t *)take(&c, 2 * a->n_presyn);
    if (!c.ok) return false;

    if (!find_section(d, size, n_sections, "CHEMOMAP", &c)) return false;
    for (int i = 0; i < WM_N_PC; i++) {
        const uint8_t *p = (const uint8_t *)take(&c, 4);
        if (!p) return false;
        memcpy(&a->chemo_pair[i][0], p, 2);
        memcpy(&a->chemo_pair[i][1], p + 2, 2);
    }

    if (!find_section(d, size, n_sections, "STIMSETS", &c)) return false;
    a->n_hunger = take_u32(&c);
    a->hunger_idx = (const uint16_t *)take(&c, 2 * a->n_hunger);
    a->n_nose = take_u32(&c);
    a->nose_idx = (const uint16_t *)take(&c, 2 * a->n_nose);
    a->n_food = take_u32(&c);
    a->food_idx = (const uint16_t *)take(&c, 2 * a->n_food);
    if (!c.ok) return false;

    if (!find_section(d, size, n_sections, "CORPUS", &c)) return false;
    a->n_sentences = take_u32(&c);
    a->n_tokens = take_u32(&c);
    a->sent_start = (const uint32_t *)take(&c, 4 * (a->n_sentences + 1));
    a->tok_vocab = (const uint16_t *)take(&c, 2 * a->n_tokens);
    a->tok_pos = (const uint8_t *)take(&c, a->n_tokens);
    a->tok_len = (const uint8_t *)take(&c, a->n_tokens);
    a->sent_edible = (const uint8_t *)take(&c, a->n_sentences);
    align_to(&c, 8);
    take_strtab(&c, &a->tok_text);
    if (!c.ok || a->tok_text.count != a->n_tokens) return false;

    if (find_section(d, size, n_sections, "SEED", &c)) a->seed = take_u32(&c);

    return true;
}
