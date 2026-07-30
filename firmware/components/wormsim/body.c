// sim/worm.py — the trailing IK chain and the steered head.

#include "wm_math.h"
#include "wormsim.h"

#include <string.h>

void wm_body_init(wm_body *w, double ox, double oy, wm_rng *r) {
    memset(w, 0, sizeof(*w));
    w->target_x = ox;
    w->target_y = oy;

    // IKChain(facing=None): four uniform(-1, 1) per link, in this exact order.
    // 200 links = 800 draws, and they come before rand_excite's 40.
    double x = ox, y = oy;
    for (int i = 0; i < WM_N_SEGMENTS; i++) {
        double hx = x + wm_rng_uniform(r, -1.0, 1.0);
        double hy = y + wm_rng_uniform(r, -1.0, 1.0);
        double tx = hx + wm_rng_uniform(r, -1.0, 1.0);
        double ty = hy + wm_rng_uniform(r, -1.0, 1.0);
        w->hx[i] = hx; w->hy[i] = hy;
        w->tx[i] = tx; w->ty[i] = ty;
        x = tx; y = ty;
    }
}

void wm_body_consume_motor(wm_body *w, double accum_left, double accum_right) {
    const double scaling = 20.0;
    double new_dir = (accum_left - accum_right) / scaling;
    w->target_dir = w->facing_dir + new_dir * WM_PI;
    w->target_speed = (wm_fabs(accum_left) + wm_fabs(accum_right)) / (scaling * 5.0);
    w->speed_change = (w->target_speed - w->speed) / (scaling * 1.5);
}

static void chain_update(wm_body *w, double target_x, double target_y) {
    double *hx = w->hx, *hy = w->hy, *tx = w->tx, *ty = w->ty;

    hx[0] = target_x;
    hy[0] = target_y;
    // hx[1:] = tx[:-1] — each segment's head becomes the previous segment's
    // PREVIOUS-tick tail. Distinct arrays, so no aliasing to worry about.
    for (int i = WM_N_SEGMENTS - 1; i >= 1; i--) {
        hx[i] = tx[i - 1];
        hy[i] = ty[i - 1];
    }

    const double size = WM_SEGMENT_SIZE;
    const double strength = 0.998;
    for (int i = 0; i < WM_N_SEGMENTS; i++) {
        double dx = hx[i] - tx[i];
        double dy = hy[i] - ty[i];
        double dist = wm_sqrt(dx * dx + dy * dy);
        if (dist == 0.0) dist = 1e-9;  // mirrors the scalar `... or 1e-9`
        double force = (0.5 - (size / dist) * 0.5) * 0.99;
        double fx = force * dx;
        double fy = force * dy;
        tx[i] += fx * strength * 2.0;
        ty[i] += fy * strength * 2.0;
        hx[i] -= fx * (1.0 - strength) * 2.0;
        hy[i] -= fy * (1.0 - strength) * 2.0;
    }
}

void wm_body_step(wm_body *w) {
    w->speed += w->speed_change;
    // Smallest signed angle facing->target. Python's % is floored, so this is
    // wm_pymod, not fmod — the sign differs for negative operands.
    double diff = wm_pymod(w->target_dir - w->facing_dir + WM_PI, 2.0 * WM_PI) - WM_PI;
    if (diff > 0.0) w->facing_dir += 0.1;
    else if (diff < 0.0) w->facing_dir -= 0.1;

    // worm-sim used screen coords (y down): target.y -= sin(facing) * speed.
    // v7 kept the sign and lets the renderer decide which way y points.
    w->target_x += wm_cos(w->facing_dir) * w->speed;
    w->target_y -= wm_sin(w->facing_dir) * w->speed;

    chain_update(w, w->target_x, w->target_y);
}
