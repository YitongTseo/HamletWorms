// HamletWorms — one worm from wordswordsworms.org, living on the desk.
//
// Everything is local. The connectome, the corpus, the learned sense of taste
// and every spoken word are in flash; there is no network and no SD card. Power
// it and the animal starts crawling.
//
// Two tasks:
//   worm  (core 1) — sim at 60 Hz plus the renderer, which is the expensive half
//   voice (core 0) — ADPCM out of flash into the ES8311, so the sim never waits

#include <string.h>

// esp-bsp.h first: bsp/display.h declares esp_err_t-returning functions without
// including esp_err.h itself, so it only compiles behind the umbrella header.
#include "bsp/esp-bsp.h"
#include "bsp/display.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_ops.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "driver/spi_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "voice.h"
#include "wormrender.h"
#include "wormsim.h"

static const char *TAG = "worm";

static esp_lcd_panel_handle_t s_panel;
static esp_lcd_panel_io_handle_t s_io;

// esp_lcd_panel_draw_bitmap only QUEUES the transfer; the DMA reads the buffer
// afterwards. The renderer was handing it a band and immediately overwriting
// that same buffer with the next one, and after ~100 frames it wedged inside
// draw_bitmap waiting on a transaction result that never came.
//
// So: one band in flight at a time, and the buffer is not touched again until
// the panel says it is done with it.
static SemaphoreHandle_t s_blit_done;

static bool IRAM_ATTR on_color_done(esp_lcd_panel_io_handle_t io,
                                    esp_lcd_panel_io_event_data_t *ev, void *ctx) {
    BaseType_t hp = pdFALSE;
    xSemaphoreGiveFromISR(s_blit_done, &hp);
    return hp == pdTRUE;
}
static wm_asset s_asset;
static wm_world *s_world;
static wr_ctx s_ctx;
static void *s_world_storage;
static uint32_t s_generation;
static char s_title[32], s_subtitle[64];
static volatile uint32_t s_frames_total;
// Where the worm task last was. The heartbeat prints it, so a hang says which
// phase it hung in instead of just going quiet.
static volatile int s_stage;      // 0 sim, 1 draw, 2 blit
static volatile int s_stage_band;
static volatile uint32_t s_blit_timeouts;
static bool s_blit_pending;

// Pull two fields out of the asset's META blob without dragging in a JSON
// parser for a string the bake tool wrote itself.
static void meta_field(const char *json, const char *key, char *out, size_t cap) {
    out[0] = 0;
    if (!json) return;
    char pat[40];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return;
    p = strchr(p + strlen(pat), ':');
    if (!p) return;
    while (*p && *p != '"') p++;
    if (!*p) return;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
    out[i] = 0;
}

// Rows per band. Two independent reasons this is not a whole frame:
//
//  - A 466x466 frame is 434 KB, far over the SPI bus's 32 KB max transaction,
//    so esp_lcd splits it and sets SPI_TRANS_CS_KEEP_ACTIVE on every chunk but
//    the last. That flag is only legal once the bus has been acquired with
//    spi_device_acquire_bus(), which esp_lcd never does, so every frame came
//    back "spi transmit (queue) color failed".
//  - The rasteriser's scratch fits in internal SRAM at this size. Drawing into
//    a PSRAM framebuffer cost 286-475 ms a frame against 28 ms of transfer.
//
// Queried from the bus at startup rather than guessed, then rounded down to an
// even count because the panel wants even row boundaries.
static int s_band_rows = 32;

static void blit_band(void *user, int y, int h, const uint16_t *pixels) {
    s_stage = 2;
    s_stage_band = y / (h ? h : 1);
    // Wait for the PREVIOUS band before starting this one, not for this one
    // after. The renderer has already moved to the other buffer, so the panel
    // is free to transmit band N while band N+1 is being drawn.
    // Bounded, not portMAX_DELAY: a lost completion should cost one dropped
    // frame, not the whole animal.
    if (s_blit_pending && xSemaphoreTake(s_blit_done, pdMS_TO_TICKS(200)) != pdTRUE)
        s_blit_timeouts++;
    esp_lcd_panel_draw_bitmap(s_panel, 0, y, WR_W, y + h, pixels);
    s_blit_pending = true;
    s_stage = 1;
}

// The IK chain starts as random jitter (IKChain with facing=None draws 800
// uniforms) and needs a moment to relax into an animal. Skipping it on the host
// looks like a knot of string for the first second.
#define WARMUP_TICKS 600

static bool mount_worm(void) {
    const esp_partition_t *p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, WORM_PARTITION_SUBTYPE, "worm");
    if (!p) {
        ESP_LOGE(TAG, "no `worm` partition");
        return false;
    }
    const void *map = NULL;
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &map, &h) != ESP_OK) {
        ESP_LOGE(TAG, "worm mmap failed");
        return false;
    }
    if (!wm_asset_open(&s_asset, map, p->size)) {
        ESP_LOGE(TAG, "worm asset not flashed or corrupt");
        return false;
    }
    ESP_LOGI(TAG, "asset: %lu vocab, %lu neurons, %lu edges, %lu sentences, seed %lu",
             (unsigned long)s_asset.n_vocab, (unsigned long)s_asset.n_neurons,
             (unsigned long)s_asset.n_edges, (unsigned long)s_asset.n_sentences,
             (unsigned long)s_asset.seed);
    if (s_asset.meta) ESP_LOGI(TAG, "meta: %s", s_asset.meta);
    return true;
}

static void worm_task(void *arg) {
    int64_t next_tick_us = esp_timer_get_time();
    int64_t fps_t0 = next_tick_us;
    int frames = 0;
    int64_t us_sim = 0, us_draw = 0, us_blit = 0;
    int64_t last_us = next_tick_us;
    float s_hold = 2.5f;

    for (;;) {
        // Sim is authoritative on timing: 60 Hz, exactly as World.tick assumes.
        // The renderer takes whatever is left.
        s_stage = 0;
        int64_t t_loop0 = esp_timer_get_time();
        int64_t now = t_loop0;
        int ticks = 0;
        // Cap at a second of catch-up so a hitch cannot spiral, but high enough
        // that the sim holds a true 60 Hz: at 8 the worm ran in slow motion,
        // ~15 ticks/s, because the render frame rate was gating it.
        while (now >= next_tick_us && ticks < 64) {
            wm_world_tick(s_world);

            wm_eaten got[WM_EATEN_CAP];
            int n = wm_world_drain_eaten(s_world, got, WM_EATEN_CAP);
            for (int i = 0; i < n; i++) {
                uint32_t len;
                const char *s = wm_str(&s_asset.tok_text, got[i].tok, &len);
                ESP_LOGI(TAG, "ate %.*s", (int)len, s);
                voice_say(s_asset.tok_vocab[got[i].tok]);
                s_ctx.flash = 1.0f;  // neurons fire, ring leaves the head
            }

            // One pass of the play is one generation: 5498 sentences at a
            // 4.5 s spawn interval, about 6.9 hours. The scroller does not loop
            // (that is what the server does when it is evolving), so without
            // this the worm would spend the rest of its life crawling an empty
            // field. Re-seeding with the same seed replays the same life
            // exactly, which is what a board with no network should do until
            // new weights arrive.
            if (wm_scroller_exhausted(&s_world->scroller)) {
                s_generation++;
                ESP_LOGI(TAG, "corpus exhausted at tick %lld; beginning pass %lu",
                         (long long)s_world->tick_count, (unsigned long)s_generation + 1);
                wm_world_init(s_world, &s_asset, s_asset.seed, s_world_storage);
                for (int k = 0; k < WARMUP_TICKS; k++) wm_world_tick(s_world);
                wm_eaten drop[WM_EATEN_CAP];
                wm_world_drain_eaten(s_world, drop, WM_EATEN_CAP);
                s_ctx.title_alpha = 1.0f;   // name itself again
                s_hold = 2.5f;
                next_tick_us = esp_timer_get_time();
                break;
            }

            next_tick_us += 1000000 / WM_BODY_TICK_HZ;
            ticks++;
        }

        // Decay on elapsed time, not per frame, so the flash lasts the same
        // ~0.2 s whether the renderer is managing 10 fps or 20.
        float dt = (float)(t_loop0 - last_us) / 1e6f;
        last_us = t_loop0;
        s_ctx.flash -= s_ctx.flash * dt * 3.2f;  // ~0.3 s tail
        // Hold the identity card for a couple of seconds, then let it go.
        if (s_ctx.title_alpha > 0.0f) {
            s_hold -= dt;
            if (s_hold < 0.0f) s_ctx.title_alpha -= dt * 0.7f;
            if (s_ctx.title_alpha < 0.0f) s_ctx.title_alpha = 0.0f;
        }
        if (s_ctx.flash < 0.002f) s_ctx.flash = 0.0f;

        int64_t t_a = esp_timer_get_time();
        s_stage = 1;
        wr_draw_banded(&s_ctx, s_world, blit_band, NULL);
        // Drain the final band before the next frame reuses its buffer.
        if (s_blit_pending) {
            if (xSemaphoreTake(s_blit_done, pdMS_TO_TICKS(200)) != pdTRUE) s_blit_timeouts++;
            s_blit_pending = false;
        }
        int64_t t_b = esp_timer_get_time();
        // Draw and blit are interleaved per band now, so they are timed as one.
        us_draw += t_b - t_a;
        us_sim += t_a - t_loop0;

        s_frames_total++;
        if (++frames == 60) {
            int64_t t = esp_timer_get_time();
            ESP_LOGI(TAG,
                     "%.1f fps | sim %.0f  draw %.0f ms | tick %lld | "
                     "stack worm %u voice %lu | dropped %lu | blit-to %lu",
                     60.0 * 1e6 / (double)(t - fps_t0),
                     us_sim / 60000.0, us_draw / 60000.0,
                     (long long)s_world->tick_count,
                     (unsigned)uxTaskGetStackHighWaterMark(NULL),
                     (unsigned long)voice_stack_free(),
                     (unsigned long)voice_dropped(),
                     (unsigned long)s_blit_timeouts);
            fps_t0 = t;
            us_sim = us_draw = us_blit = 0;
            frames = 0;
        }
        vTaskDelay(1);  // let the idle task run so the watchdog stays happy
    }
}

// Independent of the render loop, so it can tell "the worm task stopped" apart
// from "the USB-JTAG console stopped accepting writes" — the log goes quiet
// after ~11 s and those two look identical from the host end.
static void heartbeat_task(void *arg) {
    for (uint32_t i = 0;; i++) {
        ESP_LOGI("beat", "%lu | tick %lld | frames %lu | stage %d band %d",
                 (unsigned long)i, (long long)(s_world ? s_world->tick_count : 0),
                 (unsigned long)s_frames_total, s_stage, s_stage_band);
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void) {
    ESP_ERROR_CHECK(bsp_i2c_init());

    // Raw panel handle rather than bsp_display_start(): the renderer produces a
    // whole 466x466 frame itself, so LVGL would only add a copy.
    esp_lcd_panel_io_handle_t io = NULL;
    ESP_ERROR_CHECK(bsp_display_new(NULL, &s_panel, &io));
    s_io = io;
    s_blit_done = xSemaphoreCreateBinary();
    const esp_lcd_panel_io_callbacks_t cbs = {.on_color_trans_done = on_color_done};
    ESP_ERROR_CHECK(esp_lcd_panel_io_register_event_callbacks(s_io, &cbs, NULL));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(s_panel, true));
    bsp_display_brightness_init();
    bsp_display_brightness_set(100);

    size_t max_trans = 0;
    if (spi_bus_get_max_transaction_len(BSP_LCD_SPI_NUM, &max_trans) == ESP_OK && max_trans) {
        int rows = (int)(max_trans / (WR_W * sizeof(uint16_t)));
        if (rows < 1) rows = 1;
        if (rows > 32) rows = 32;   // keeps the scratch inside internal SRAM
        if (rows > 1) rows &= ~1;   // even: the panel wants even row boundaries
        s_band_rows = rows;
    }
    ESP_LOGI(TAG, "spi max transaction %u B -> %d rows per band (%d bands)",
             (unsigned)max_trans, s_band_rows, (WR_H + s_band_rows - 1) / s_band_rows);

    if (!mount_worm()) return;

    // The world is ~30 KB now that the scroller is sized for the real corpus,
    // so it goes in internal SRAM: the IK chain rewrites every segment 60 times
    // a second and the connectome scatters across psyn on every brain tick.
    // Fall back to PSRAM rather than refuse to boot.
    s_world = heap_caps_malloc(sizeof(wm_world), MALLOC_CAP_INTERNAL);
    if (!s_world) s_world = heap_caps_malloc(sizeof(wm_world), MALLOC_CAP_SPIRAM);
    void *storage = heap_caps_malloc(wm_world_bytes(&s_asset), MALLOC_CAP_INTERNAL);
    if (!storage) storage = heap_caps_malloc(wm_world_bytes(&s_asset), MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "world %u B + %u B state", (unsigned)sizeof(wm_world),
             (unsigned)wm_world_bytes(&s_asset));

    // The band scratch must NOT. It takes scattered single-byte writes in the
    // coverage pass, which is exactly what PSRAM is worst at, and it is handed
    // to the SPI DMA engine. Internal SRAM, DMA-capable.
    uint8_t *scratch = heap_caps_malloc(wr_scratch_bytes(s_band_rows),
                                        MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);

    // The graticule is built once and only ever read sequentially, so PSRAM is
    // the right home for its 217 KB.
    uint8_t *globe = heap_caps_malloc(wr_globe_bytes(), MALLOC_CAP_SPIRAM);
    if (!s_world || !storage || !scratch || !globe) {
        ESP_LOGE(TAG, "out of PSRAM");
        return;
    }

    s_world_storage = storage;
    wm_world_init(s_world, &s_asset, s_asset.seed, storage);
    wr_build_globe(globe);
    wr_init(&s_ctx, scratch, s_band_rows, globe);

    meta_field(s_asset.meta, "worm", s_title, sizeof(s_title));
    char flask[24], gen[24];
    meta_field(s_asset.meta, "flask", flask, sizeof(flask));
    meta_field(s_asset.meta, "gen", gen, sizeof(gen));
    snprintf(s_subtitle, sizeof(s_subtitle), "%s  %s", flask, gen);
    s_ctx.title = s_title;
    s_ctx.subtitle = s_subtitle;
    s_ctx.title_alpha = 1.0f;

    ESP_LOGI(TAG, "settling the body (%d ticks)...", WARMUP_TICKS);
    for (int i = 0; i < WARMUP_TICKS; i++) wm_world_tick(s_world);
    wm_eaten discard[WM_EATEN_CAP];
    wm_world_drain_eaten(s_world, discard, WM_EATEN_CAP);  // don't speak the warmup

    // BSP_I2S_DUPLEX_MONO_CFG is private to the BSP's .c, so hand it NULL for
    // the default channel setup; esp_codec_dev_open then drives the real rate
    // through the codec's data interface, which reclocks I2S to match.
    ESP_ERROR_CHECK(bsp_audio_init(NULL));
    esp_codec_dev_handle_t spk = bsp_audio_codec_speaker_init();
    if (spk) {
        // 8 kHz mono, matching the bank — nothing resamples on the way out.
        esp_codec_dev_sample_info_t fs = {
            .sample_rate = 8000, .channel = 1, .bits_per_sample = 16,
        };
        esp_codec_dev_open(spk, &fs);
        esp_codec_dev_set_out_vol(spk, 100);  // plus 2.5x digital gain in voice.c
        if (voice_init(spk)) voice_start();
    } else {
        ESP_LOGW(TAG, "no speaker; the worm will eat in silence");
    }

    xTaskCreatePinnedToCore(worm_task, "worm", 8192, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(heartbeat_task, "beat", 3072, NULL, 2, NULL, 0);
}
