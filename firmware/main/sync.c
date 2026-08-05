// Pulling new worm genomes off the network.
//
// The server that runs the flasks already commits every generation to GitHub —
// v7/data/poetry-2/generations/<flask>/gen-NNNN/<worm>/weights.json — and that
// repo is public, so raw.githubusercontent.com is the whole distribution
// mechanism. No endpoint had to be added to the server, and nothing about the
// wall is centralised: each board is a client of a git history.
//
// The board probes forward from the generation it is running. gen-NNNN either
// exists or 404s, and 404 is the answer "you are up to date" — cheaper and less
// fragile than asking the API to list a directory, which is rate limited.
//
// Everything here is best-effort. No wifi, no route, no such generation,
// malformed JSON — the worm carries on with the genome baked into its flash.
// A board that cannot reach the internet is still a complete animal.

#include "sync.h"

#include <stdlib.h>
#include <string.h>

#include "cJSON.h"
#include "esp_crt_bundle.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"

static const char *TAG = "sync";

#define GENOME_URL_FMT \
    "https://raw.githubusercontent.com/YitongTseo/HamletRNAWorld/main/" \
    "v7/data/poetry-2/generations/%s/gen-%04lu/%s/weights.json"

// weights.json for one worm is ~105 KB. Give it room and refuse anything wilder
// than that rather than growing without bound on a bad response.
#define MAX_GENOME_BYTES (400 * 1024)

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAILED_BIT BIT1

static EventGroupHandle_t s_wifi_events;
static int s_retries;
static sync_cfg_t s_cfg;

// ---------------------------------------------------------------------------
// wifi
// ---------------------------------------------------------------------------

static void on_wifi(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retries < 5) {
            s_retries++;
            esp_wifi_connect();
        } else {
            xEventGroupSetBits(s_wifi_events, WIFI_FAILED_BIT);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        s_retries = 0;
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
    }
}

static bool wifi_up(void) {
    if (!s_cfg.ssid[0]) {
        ESP_LOGI(TAG, "no wifi configured; staying offline");
        return false;
    }
    s_wifi_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                        on_wifi, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                        on_wifi, NULL, NULL));
    wifi_config_t wc = {0};
    strncpy((char *)wc.sta.ssid, s_cfg.ssid, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, s_cfg.pass, sizeof(wc.sta.password) - 1);
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());

    EventBits_t bits = xEventGroupWaitBits(s_wifi_events, WIFI_CONNECTED_BIT | WIFI_FAILED_BIT,
                                           pdFALSE, pdFALSE, pdMS_TO_TICKS(20000));
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "wifi up on \"%s\"", s_cfg.ssid);
        return true;
    }
    ESP_LOGW(TAG, "wifi did not come up; the worm runs on its baked genome");
    return false;
}

// ---------------------------------------------------------------------------
// fetch
// ---------------------------------------------------------------------------

typedef struct {
    char *buf;
    int len, cap;
} fetch_buf;

static esp_err_t on_http(esp_http_client_event_t *e) {
    if (e->event_id != HTTP_EVENT_ON_DATA) return ESP_OK;
    fetch_buf *fb = (fetch_buf *)e->user_data;
    if (!fb || fb->len + e->data_len > fb->cap) return ESP_OK;  // truncate, caught later
    memcpy(fb->buf + fb->len, e->data, e->data_len);
    fb->len += e->data_len;
    return ESP_OK;
}

// Returns the body length, 0 on 404, negative on error. `out` is PSRAM.
static int fetch(const char *url, char *out, int cap) {
    fetch_buf fb = {.buf = out, .len = 0, .cap = cap};
    esp_http_client_config_t cfg = {
        .url = url,
        .event_handler = on_http,
        .user_data = &fb,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .timeout_ms = 20000,
        .buffer_size = 4096,
    };
    esp_http_client_handle_t cl = esp_http_client_init(&cfg);
    if (!cl) return -1;
    esp_err_t err = esp_http_client_perform(cl);
    int status = esp_http_client_get_status_code(cl);
    esp_http_client_cleanup(cl);
    if (err != ESP_OK) return -1;
    if (status == 404) return 0;
    if (status != 200) return -1;
    return fb.len;
}

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------

// The neuron table is sorted (bake.py emits sorted(neurons)), so names resolve
// by binary search rather than by scanning 396 entries per edge.
static int neuron_index(const wm_asset *a, const char *name) {
    int lo = 0, hi = (int)a->n_neurons - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        uint32_t len;
        const char *nm = wm_str(&a->neurons, (uint32_t)mid, &len);
        int c = strncmp(name, nm, len);
        if (c == 0 && name[len] != '\0') c = 1;  // name is longer: sorts after
        if (c == 0) return mid;
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

// Fill `out` (n_edges, CSR order) from the JSON. Deliberately resolves every
// edge by NAME rather than trusting the file to enumerate in the same order
// bake.py saw — the topology is fixed, but the order a dict serialises in is
// not something to bet an animal on.
static bool parse_weights(const wm_asset *a, const char *json, int len, double *out) {
    cJSON *root = cJSON_ParseWithLength(json, (size_t)len);
    if (!root) {
        ESP_LOGE(TAG, "genome is not valid JSON");
        return false;
    }
    // Start from the baked weights so an edge the file omits keeps its value
    // rather than silently becoming zero.
    memcpy(out, a->syn_w, sizeof(double) * a->n_edges);

    int matched = 0, missed = 0;
    cJSON *pre = NULL;
    cJSON_ArrayForEach(pre, root) {
        int pi = neuron_index(a, pre->string);
        if (pi < 0) { missed++; continue; }
        uint32_t s = a->row_start[pi], e = a->row_start[pi + 1];
        cJSON *post = NULL;
        cJSON_ArrayForEach(post, pre) {
            int qi = neuron_index(a, post->string);
            if (qi < 0) { missed++; continue; }
            bool found = false;
            for (uint32_t k = s; k < e; k++) {
                if (a->syn_col[k] == (uint16_t)qi) {
                    out[k] = cJSON_GetNumberValue(post);
                    matched++;
                    found = true;
                    break;
                }
            }
            if (!found) missed++;
        }
    }
    cJSON_Delete(root);

    // The connectome's topology is fixed across generations — only the weights
    // evolve — so anything other than a full match means this file is not the
    // genome we think it is.
    if (matched != (int)a->n_edges || missed) {
        ESP_LOGE(TAG, "genome shape mismatch: %d/%lu edges matched, %d unresolved",
                 matched, (unsigned long)a->n_edges, missed);
        return false;
    }
    ESP_LOGI(TAG, "genome parsed: %d edges", matched);
    return true;
}

// ---------------------------------------------------------------------------
// the task
// ---------------------------------------------------------------------------

static void sync_task(void *arg) {
    if (!wifi_up()) vTaskDelete(NULL);

    char *body = heap_caps_malloc(MAX_GENOME_BYTES, MALLOC_CAP_SPIRAM);
    double *weights = heap_caps_malloc(sizeof(double) * s_cfg.asset->n_edges,
                                       MALLOC_CAP_SPIRAM);
    if (!body || !weights) {
        ESP_LOGE(TAG, "no room to sync");
        vTaskDelete(NULL);
    }

    for (;;) {
        uint32_t epoch = s_cfg.epoch;
        uint32_t newest = epoch;
        // Probe forward. Each generation the server finishes is one more
        // directory in the history, so walking until a 404 finds the head
        // without asking anything to enumerate.
        for (uint32_t g = epoch + 1; g <= epoch + 64; g++) {
            char url[320];
            snprintf(url, sizeof(url), GENOME_URL_FMT, s_cfg.flask, (unsigned long)g,
                     s_cfg.worm);
            int n = fetch(url, body, MAX_GENOME_BYTES);
            if (n == 0) break;       // 404: that is the end of the history
            if (n < 0) { newest = epoch; break; }  // network trouble; try later
            if (parse_weights(s_cfg.asset, body, n, weights)) newest = g;
        }

        if (newest != epoch) {
            ESP_LOGI(TAG, "gen-%04lu -> gen-%04lu", (unsigned long)epoch,
                     (unsigned long)newest);
            s_cfg.epoch = newest;
            nvs_handle_t nvs;
            if (nvs_open("worm", NVS_READWRITE, &nvs) == ESP_OK) {
                nvs_set_u32(nvs, "epoch", newest);
                nvs_commit(nvs);
                nvs_close(nvs);
            }
            s_cfg.on_genome(weights, newest, s_cfg.ctx);
        }

        vTaskDelay(pdMS_TO_TICKS(s_cfg.poll_minutes * 60 * 1000));
    }
}

void sync_start(const sync_cfg_t *cfg) {
    s_cfg = *cfg;

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    // A board that has already caught up should not re-download the whole
    // history every time it is power cycled.
    nvs_handle_t nvs;
    if (nvs_open("worm", NVS_READONLY, &nvs) == ESP_OK) {
        uint32_t stored = 0;
        if (nvs_get_u32(nvs, "epoch", &stored) == ESP_OK && stored > s_cfg.epoch)
            s_cfg.epoch = stored;
        nvs_close(nvs);
    }
    ESP_LOGI(TAG, "%s/%s at gen-%04lu, polling every %d min", s_cfg.flask, s_cfg.worm,
             (unsigned long)s_cfg.epoch, s_cfg.poll_minutes);

    // Low priority and its own core: the animal must never stutter for this.
    xTaskCreatePinnedToCore(sync_task, "sync", 8192, NULL, 3, NULL, 0);
}
