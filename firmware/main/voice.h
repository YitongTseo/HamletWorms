#ifndef VOICE_H
#define VOICE_H

#include <stdbool.h>
#include <stdint.h>

#include "esp_codec_dev.h"

// Custom data-partition subtype, matching partitions.csv.
#define VOICE_PARTITION_SUBTYPE 0x41
#define WORM_PARTITION_SUBTYPE 0x40

bool voice_init(esp_codec_dev_handle_t codec);
void voice_start(void);

// Queue a word by vocabulary id — the same integer wm_asset.tok_vocab holds.
// 0xFFFF (OOV) is silent. Never blocks; drops when the queue is full.
void voice_say(uint16_t vocab_id);

#endif
