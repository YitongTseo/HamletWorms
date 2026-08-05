#!/usr/bin/env bash
# The S3's native USB re-enumerates under a different name after a replug
# (usbmodem101 vs usbmodem1101), so nothing should hard-code it.
ls /dev/cu.usbmodem* 2>/dev/null | head -1
