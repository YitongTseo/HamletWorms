// wm_math — the swappable math layer.
//
// Everything the sim needs that is NOT bit-exactly specified by IEEE 754 lives
// here, so the Phase 2 "lock to the server" work is confined to one file.
//
// Bit-exact already, on any conforming platform:
//   + - * /  and sqrt   — IEEE 754 specifies these to be correctly rounded.
//
// NOT bit-exact across platforms, and therefore isolated below:
//   sin cos atan2 hypot exp — libm implementations (glibc vs newlib vs
//   Apple libm) agree to well under 1 ULP but not to the last bit.
//
// Why one ULP is not "close enough": Connectome.run() fires on `psyn > 30`, a
// hard discontinuity. A single flipped neuron sends the body down a different
// path, and a generation is a full pass of Hamlet (5498 sentences x 4.5 s spawn
// ~= 6.9 hours of sim time). Divergence is guaranteed on that timescale.
//
// Phase 2 replaces these with fixed polynomial kernels compiled into BOTH this
// firmware and the Python sim (via a small extension module), plus a harness
// that runs 100k ticks on each and diffs a state hash. Until then the board is
// an independent worm of the same genome rather than a lock-step replica —
// visually indistinguishable, but its poem is its own.
//
// Also not yet exact: wm_dot() sums left to right, while numpy's BLAS gemv
// blocks and may fuse multiply-add. Same fix, same phase.

#ifndef WM_MATH_H
#define WM_MATH_H

#include <math.h>

#define WM_PI 3.141592653589793

#define wm_sin sin
#define wm_cos cos
#define wm_atan2 atan2
#define wm_hypot hypot
#define wm_exp exp
#define wm_sqrt sqrt
#define wm_fabs fabs

// Python's % on floats is FLOORED; C's fmod is TRUNCATED. They differ in sign
// for negative operands, and both `(target_dir - facing_dir + pi) % (2*pi)` in
// WormBody.step and `(food_angle - facing_dir + pi) % (2*pi)` in
// World._compute_smells depend on the Python behaviour.
static inline double wm_pymod(double x, double y) {
    double r = fmod(x, y);
    if (r != 0.0 && ((r < 0.0) != (y < 0.0))) r += y;
    return r;
}

static inline double wm_relu(double x) { return x > 0.0 ? x : 0.0; }

// _sigmoid in server/embedding.py clips to [-60, 60] before exp.
static inline double wm_sigmoid(double x) {
    if (x < -60.0) x = -60.0;
    else if (x > 60.0) x = 60.0;
    return 1.0 / (1.0 + wm_exp(-x));
}

// y[j] = sum_k x[k] * W[k*n_out + j] + b[j]   (row-major W, matching numpy)
static inline void wm_affine(const double *x, const double *W, const double *b,
                             int n_in, int n_out, double *y) {
    for (int j = 0; j < n_out; j++) {
        double s = 0.0;
        for (int k = 0; k < n_in; k++) s += x[k] * W[k * n_out + j];
        y[j] = s + b[j];
    }
}

#endif  // WM_MATH_H
