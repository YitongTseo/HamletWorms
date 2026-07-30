// sim/text_scroller.py — Hamlet lines drifting upward.
//
// The Python keeps a `_dead` set keyed by (line_id, word_idx) that outlives the
// line; here the eaten flag lives on the word itself, which is equivalent
// because line_ids are never reused and alive_words() only ever walks _active.

#include "wm_math.h"
#include "wormsim.h"

#include <string.h>

void wm_scroller_init(wm_scroller *s, const wm_asset *a, bool loop) {
    memset(s, 0, sizeof(*s));
    s->a = a;
    s->loop = loop;
}

bool wm_scroller_exhausted(const wm_scroller *s) {
    return !s->loop && s->sent_idx >= s->a->n_sentences && s->n_lines == 0;
}

static void spawn(wm_scroller *s) {
    const wm_asset *a = s->a;
    uint32_t si;
    if (s->loop) {
        si = s->sent_idx % a->n_sentences;
    } else {
        if (s->sent_idx >= a->n_sentences) return;
        si = s->sent_idx;
    }
    s->sent_idx++;

    uint32_t start = a->sent_start[si], end = a->sent_start[si + 1];
    uint32_t n = end - start;
    if (n > WM_MAX_LINE_WORDS) n = WM_MAX_LINE_WORDS;
    if (s->n_lines >= WM_MAX_ACTIVE_LINES) return;

    wm_line *L = &s->lines[s->n_lines++];
    L->line_id = s->line_id++;
    L->sent_idx = si;
    L->edible = a->sent_edible[si] != 0;
    L->n_words = (uint16_t)n;

    // total_w = sum(len(t) * CHAR_W) + WORD_GAP * (n - 1), summed left to right.
    // len() is code points; bake.py precomputed that in tok_len.
    double total_w = 0.0;
    for (uint32_t i = 0; i < n; i++) total_w += (double)a->tok_len[start + i] * WM_CHAR_W;
    total_w += WM_WORD_GAP * (double)((int)n - 1);

    double x = WM_CENTER_X - total_w / 2.0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t tok = start + i;
        double cw = (double)a->tok_len[tok] * WM_CHAR_W;
        L->w[i].tok = tok;
        L->w[i].x = x + cw / 2.0;  // centre of the word
        L->w[i].y = WM_SPAWN_Y;
        L->w[i].eaten = false;
        x += cw + WM_WORD_GAP;
    }
}

void wm_scroller_step(wm_scroller *s, double dt) {
    s->elapsed += dt;

    for (int i = 0; i < s->n_lines; i++)
        for (uint16_t j = 0; j < s->lines[i].n_words; j++)
            s->lines[i].w[j].y -= WM_SCROLL_SPEED * dt;

    // Drop lines with no word still above KILL_Y, preserving order.
    int keep = 0;
    for (int i = 0; i < s->n_lines; i++) {
        bool any = false;
        for (uint16_t j = 0; j < s->lines[i].n_words; j++)
            if (s->lines[i].w[j].y > WM_KILL_Y) { any = true; break; }
        if (any) {
            if (keep != i) s->lines[keep] = s->lines[i];
            keep++;
        }
    }
    s->n_lines = keep;

    if (s->elapsed >= s->next_spawn) {
        if (s->loop || s->sent_idx < s->a->n_sentences) spawn(s);
        s->next_spawn = s->elapsed + WM_SPAWN_INTERVAL;
    }
}
