"""
Thin wrapper around demucs that patches torchaudio.save to use soundfile,
working around the torchcodec/FFmpeg shared-library incompatibility on this system
(torchaudio 2.9.x requires torchcodec which needs libavutil.so.57-60, unavailable here).
"""

import sys
import numpy as np
import soundfile as sf


def _save_soundfile(uri, src, sample_rate, channels_first=True, **kwargs):
    wav = src.numpy()
    if channels_first:
        wav = wav.T  # (samples, channels) for soundfile
    sf.write(str(uri), wav, sample_rate)


import torchaudio

torchaudio.save = _save_soundfile

from demucs.__main__ import main

sys.exit(main() or 0)
