// CPython's random.Random, ported.
//
// Not "an MT19937" — *the* one, including CPython's init_by_array seeding path
// and the exact 53-bit double construction, because the worm's entire life is
// downstream of these draws. World.__post_init__ makes them in this order:
//
//   1. WormBody -> IKChain(facing=None) -> 4 x uniform(-1, 1) per link,
//      800 draws for the 200-segment body.
//   2. Connectome.rand_excite() -> 40 x choice(list(weights.keys())).
//
// Note that the body is built BEFORE rand_excite, even though the brain is
// constructed first — the draws happen in __post_init__ order, not field order.

#include "wormsim.h"

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfu
#define UPPER_MASK 0x80000000u
#define LOWER_MASK 0x7fffffffu

static void init_genrand(wm_rng *r, uint32_t s) {
    r->mt[0] = s;
    for (int i = 1; i < N; i++)
        r->mt[i] = (1812433253u * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) + (uint32_t)i);
    r->index = N;
}

static void init_by_array(wm_rng *r, const uint32_t *key, int klen) {
    init_genrand(r, 19650218u);
    int i = 1, j = 0;
    int k = N > klen ? N : klen;
    for (; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1664525u))
                   + key[j] + (uint32_t)j;
        i++; j++;
        if (i >= N) { r->mt[0] = r->mt[N - 1]; i = 1; }
        if (j >= klen) j = 0;
    }
    for (k = N - 1; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) * 1566083941u))
                   - (uint32_t)i;
        i++;
        if (i >= N) { r->mt[0] = r->mt[N - 1]; i = 1; }
    }
    r->mt[0] = 0x80000000u;
    r->index = N;
}

void wm_rng_seed(wm_rng *r, uint32_t seed) {
    // CPython's random_seed() takes abs(n) and splits it into 32-bit little-
    // endian words, with at least one word. Seeds here are small positive ints
    // (Liam's is 12), so that is a single-element key.
    uint32_t key[1] = {seed};
    init_by_array(r, key, 1);
}

uint32_t wm_rng_u32(wm_rng *r) {
    if (r->index >= N) {
        static const uint32_t mag01[2] = {0x0u, MATRIX_A};
        uint32_t y;
        int kk;
        for (kk = 0; kk < N - M; kk++) {
            y = (r->mt[kk] & UPPER_MASK) | (r->mt[kk + 1] & LOWER_MASK);
            r->mt[kk] = r->mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        for (; kk < N - 1; kk++) {
            y = (r->mt[kk] & UPPER_MASK) | (r->mt[kk + 1] & LOWER_MASK);
            r->mt[kk] = r->mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        y = (r->mt[N - 1] & UPPER_MASK) | (r->mt[0] & LOWER_MASK);
        r->mt[N - 1] = r->mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
        r->index = 0;
    }
    uint32_t y = r->mt[r->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

double wm_rng_double(wm_rng *r) {
    // CPython random_random(): two draws, 27 + 26 bits, exactly this expression.
    uint32_t a = wm_rng_u32(r) >> 5;
    uint32_t b = wm_rng_u32(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

double wm_rng_uniform(wm_rng *r, double a, double b) {
    return a + (b - a) * wm_rng_double(r);
}

static uint32_t getrandbits(wm_rng *r, int k) {
    if (k == 0) return 0;
    if (k >= 32) return wm_rng_u32(r);
    return wm_rng_u32(r) >> (32 - k);
}

uint32_t wm_rng_below(wm_rng *r, uint32_t n) {
    // _randbelow_with_getrandbits: rejection-sample k = n.bit_length() bits.
    if (n == 0) return 0;
    int k = 0;
    for (uint32_t v = n; v; v >>= 1) k++;
    uint32_t x = getrandbits(r, k);
    while (x >= n) x = getrandbits(r, k);
    return x;
}
