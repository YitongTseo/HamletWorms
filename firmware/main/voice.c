// Speaking. IMA-ADPCM out of flash, through the ES8311, out the onboard speaker.
//
// The bank (tools/voices.py) is indexed by vocab id — the same integer the worm
// asset uses — so eating a word and saying it needs no string lookup, just
// asset->tok_vocab[tok].
//
// Nothing streams from an SD card. The whole 4919-word vocabulary is 10.47 MB in
// the `voices` flash partition, memory-mapped, so playback is a pointer walk.

#include "voice.h"

#include <stdlib.h>
#include <string.h>

#include "esp_log.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

static const char *TAG = "voice";

// The two tables that define IMA ADPCM.
static const int16_t STEP[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767,
};
static const int8_t INDEX[16] = {-1, -1, -1, -1, 2, 4, 6, 8,
                                 -1, -1, -1, -1, 2, 4, 6, 8};

#define PCM_CHUNK 512  // samples handed to the codec at a time

// Playback gain. 1:1 — loudness is handled in the bank by tools/voices.py,
// which speech-normalises each word before encoding. Gain here can only clip:
// the words already reach ~0.97 of full scale, and a sample that wraps is a
// loud click. Left as a knob rather than deleted, since the codec volume and
// this are the only two places loudness can be touched at runtime.
#define VOICE_GAIN_NUM 1
#define VOICE_GAIN_DEN 1

struct voice {
    const uint8_t *bank;      // mmapped `voices` partition
    uint32_t n_words;
    uint32_t sample_rate;
    uint16_t block_align;
    uint16_t samples_per_block;
    const uint8_t *index;     // n_words x (u32 offset, u32 length)

    esp_codec_dev_handle_t codec;
    QueueHandle_t q;

    // Decoded PCM for one ADPCM block. Heap, NOT stack: at block_align 1024 a
    // block expands to 2041 samples = 4082 bytes, and this used to be a local
    // in speak_blocking against a 4096-byte task stack. It overflowed the first
    // time the worm ate a word and took the whole board down with a
    // LoadProhibited inside esp_codec_dev_write.
    int16_t *block;
    uint32_t block_samples;
    TaskHandle_t task;
};

// Words left unsaid, for the log — the queue drops rather than lag when the
// worm eats faster than it can speak.
static uint32_t v_dropped;

static struct voice V;

bool voice_init(esp_codec_dev_handle_t codec) {
    const esp_partition_t *p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, VOICE_PARTITION_SUBTYPE, "voices");
    if (!p) {
        ESP_LOGE(TAG, "no `voices` partition");
        return false;
    }

    const void *map = NULL;
    esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &map, &h) != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed");
        return false;
    }

    const uint8_t *b = (const uint8_t *)map;
    if (memcmp(b, "HVOX", 4) != 0) {
        ESP_LOGE(TAG, "voice bank not flashed (magic %02x%02x%02x%02x)",
                 b[0], b[1], b[2], b[3]);
        return false;
    }

    V.bank = b;
    memcpy(&V.n_words, b + 8, 4);
    memcpy(&V.sample_rate, b + 12, 4);
    uint32_t codec_id;
    memcpy(&codec_id, b + 16, 4);
    memcpy(&V.block_align, b + 20, 2);
    memcpy(&V.samples_per_block, b + 22, 2);
    V.index = b + 24;
    V.codec = codec;

    if (codec_id != 0) {
        ESP_LOGE(TAG, "bank is not ADPCM (codec=%lu)", (unsigned long)codec_id);
        return false;
    }
    ESP_LOGI(TAG, "%lu words, %lu Hz, block=%u, spb=%u",
             (unsigned long)V.n_words, (unsigned long)V.sample_rate,
             V.block_align, V.samples_per_block);

    // 1 + (block_align - 4) * 2 samples per block; a couple spare for safety.
    V.block_samples = 1 + (uint32_t)(V.block_align - 4) * 2;
    V.block = malloc(sizeof(int16_t) * (V.block_samples + 8));
    if (!V.block) {
        ESP_LOGE(TAG, "no room for the decode buffer");
        return false;
    }

    // Depth 6: the worm can eat a burst of words faster than they can be said
    // (a word averages 0.46 s). Beyond that, drop rather than fall behind — a
    // backlog of stale words is worse than a missed one.
    V.q = xQueueCreate(6, sizeof(uint16_t));
    return V.q != NULL;
}

// One IMA-ADPCM-in-WAV block: 4-byte preamble (initial predictor and step
// index), then 4-bit nibbles, low nibble first.
static int decode_block(const uint8_t *blk, int nbytes, int16_t *out) {
    int16_t pred;
    memcpy(&pred, blk, 2);
    int index = blk[2];
    int n = 0;
    out[n++] = pred;

    for (int i = 4; i < nbytes; i++) {
        uint8_t byte = blk[i];
        for (int half = 0; half < 2; half++) {
            int nib = half ? (byte >> 4) : (byte & 0x0F);
            int step = STEP[index];
            // ffmpeg's rounding, not the classic IMA reference's. The reference
            // accumulates step>>3 + step>>2 + step>>1 + step, truncating each
            // term; ffmpeg computes ((2*delta+1)*step)>>3 and truncates once.
            // Same 1.875*step, one rounding apart — and since ffmpeg encoded
            // the bank, rounding the other way drifts from the second sample on
            // and comes out as static. tools/voxcheck.py pins this down.
            int diff = ((2 * (nib & 7) + 1) * step) >> 3;
            int p = (nib & 8) ? pred - diff : pred + diff;
            if (p > 32767) p = 32767;
            else if (p < -32768) p = -32768;
            pred = (int16_t)p;
            index += INDEX[nib];
            if (index < 0) index = 0;
            else if (index > 88) index = 88;
            out[n++] = pred;
        }
    }
    return n;
}

static void speak_blocking(uint16_t vocab_id) {
    if (vocab_id >= V.n_words) return;
    uint32_t off, len;
    memcpy(&off, V.index + 8 * vocab_id, 4);
    memcpy(&len, V.index + 8 * vocab_id + 4, 4);
    if (!len) return;

    const uint8_t *p = V.bank + off;
    int16_t *block = V.block;

    for (uint32_t consumed = 0; consumed < len;) {
        int nbytes = (int)(len - consumed);
        if (nbytes > V.block_align) nbytes = V.block_align;
        int n = decode_block(p + consumed, nbytes, block);
        consumed += nbytes;

        // Saturating, never wrapping. At 1:1 this is a no-op the compiler
        // folds away; it exists so raising VOICE_GAIN_NUM stays safe.
        for (int i = 0; VOICE_GAIN_NUM != VOICE_GAIN_DEN && i < n; i++) {
            int32_t v = (int32_t)block[i] * VOICE_GAIN_NUM / VOICE_GAIN_DEN;
            block[i] = v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v);
        }
        for (int i = 0; i < n; i += PCM_CHUNK) {
            int chunk = n - i < PCM_CHUNK ? n - i : PCM_CHUNK;
            esp_codec_dev_write(V.codec, block + i, chunk * sizeof(int16_t));
        }
    }
}

static void voice_task(void *arg) {
    uint16_t id;
    for (;;) {
        if (xQueueReceive(V.q, &id, portMAX_DELAY) == pdTRUE) speak_blocking(id);
    }
}

void voice_start(void) {
    // 6 KB, not 4. The decode buffer moved to the heap, but this task also
    // descends into esp_codec_dev and the I2S driver, and 4096 was already
    // proven too tight once — the first version overflowed it and took the
    // board down the moment the worm ate its first word.
    xTaskCreatePinnedToCore(voice_task, "voice", 6144, NULL, 4, &V.task, 0);
}

void voice_say(uint16_t vocab_id) {
    if (vocab_id == 0xFFFF) return;  // OOV: punctuation and the like, silent
    // Non-blocking on purpose. The sim must not stall waiting on the speaker.
    if (xQueueSend(V.q, &vocab_id, 0) != pdTRUE) v_dropped++;
}

uint32_t voice_stack_free(void) {
    return V.task ? (uint32_t)uxTaskGetStackHighWaterMark(V.task) : 0;
}

uint32_t voice_dropped(void) { return v_dropped; }
