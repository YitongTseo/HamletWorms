// server/embedding.py forward pass, minus the 512->11 projection.
//
// tools/bake.py already folded E = ReLU(nomic512 @ W_E + b_E) into a 4919 x 11
// table, exactly as EmbeddingModel.prime() does once per generation. That is why
// the 26 MB nomic cache never has to reach the board: E depends only on the word,
// and the same table serves both the current word and the eaten history.
//
// What is left is four small matmuls. At ~20 in-range words on a 2 Hz brain tick
// that is about 60k double MACs per second.

#include "wm_math.h"
#include "wormsim.h"

#include <string.h>

// _onehot_pos: an empty history slot contributes all zeros; a real word one-hots
// its tag. Unknown tags were already folded to "X" by bake.py.
static void onehot(uint8_t tag, double *out) {
    memset(out, 0, sizeof(double) * WM_N_TAGS);
    if (tag < WM_N_TAGS) out[tag] = 1.0;
}

bool wm_chemo_embed(const wm_asset *a, uint32_t tok,
                    const uint32_t *recent, int n_recent, double *out) {
    uint16_t vi = a->tok_vocab[tok];
    if (vi == WM_OOV) return false;  // matches embed() returning None

    // --- history: zero-padded to HISTORY, most-recent-first ---
    double he[WM_D_HIST_CONCAT];
    double hpos[WM_N_TAGS * WM_HISTORY];
    for (int j = 0; j < WM_HISTORY; j++) {
        double *he_slot = he + j * WM_D_EMB;
        double *hp_slot = hpos + j * WM_N_TAGS;
        if (j < n_recent) {
            uint32_t htok = recent[j];
            uint16_t hvi = a->tok_vocab[htok];
            if (hvi == WM_OOV) memset(he_slot, 0, sizeof(double) * WM_D_EMB);
            else memcpy(he_slot, a->etable + (size_t)hvi * WM_D_EMB,
                        sizeof(double) * WM_D_EMB);
            // The POS branch tags the actual word even when the embedding
            // branch treats it as OOV — _norm strips apostrophes but
            // tag_word does not, so "o'er" tags fine and embeds as zeros.
            onehot(a->tok_pos[htok], hp_slot);
        } else {
            memset(he_slot, 0, sizeof(double) * WM_D_EMB);
            memset(hp_slot, 0, sizeof(double) * WM_N_TAGS);
        }
    }

    double hist_summary[WM_D_EMB];
    wm_affine(he, a->W_Hh, a->b_Hh, WM_D_HIST_CONCAT, WM_D_EMB, hist_summary);
    for (int i = 0; i < WM_D_EMB; i++) hist_summary[i] = wm_relu(hist_summary[i]);

    // --- embedding branch ---
    double fuse[WM_D_FUSE];
    memcpy(fuse, a->etable + (size_t)vi * WM_D_EMB, sizeof(double) * WM_D_EMB);
    memcpy(fuse + WM_D_EMB, hist_summary, sizeof(double) * WM_D_EMB);

    double emb_out[WM_D_EMB];
    wm_affine(fuse, a->W_Hf, a->b_Hf, WM_D_FUSE, WM_D_EMB, emb_out);
    for (int i = 0; i < WM_D_EMB; i++) out[i] = wm_sigmoid(emb_out[i]);

    // --- POS branch ---
    double pos_in[WM_D_POS_IN];
    onehot(a->tok_pos[tok], pos_in);
    memcpy(pos_in + WM_N_TAGS, hpos, sizeof(double) * WM_N_TAGS * WM_HISTORY);

    double pos_hid[WM_D_POS_HID];
    wm_affine(pos_in, a->W_P1, a->b_P1, WM_D_POS_IN, WM_D_POS_HID, pos_hid);
    for (int i = 0; i < WM_D_POS_HID; i++) pos_hid[i] = wm_relu(pos_hid[i]);

    double pos_out;
    wm_affine(pos_hid, a->W_P2, a->b_P2, WM_D_POS_HID, 1, &pos_out);
    out[WM_D_EMB] = wm_sigmoid(pos_out);

    return true;
}
