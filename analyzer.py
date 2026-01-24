# analyzer.py

import numpy as np
from config import (
    TOTAL_BW, TILE_BW, STEP_BW, SETTLING_READS
)
from dsp import compute_fft, freq_axis
from stitcher import SpectrumStitcher

class SpectrumAnalyzer:
    def __init__(self, sdr):
        self.sdr = sdr

    def sweep(self, center_freq):
        f_start = center_freq - TOTAL_BW / 2
        lo_freqs = np.arange(
            f_start + TILE_BW/2,
            f_start + TOTAL_BW,
            STEP_BW
        )

        stitcher = SpectrumStitcher()

        for lo in lo_freqs:
            self.sdr.tune(lo)

            for _ in range(SETTLING_READS):
                self.sdr.read()

            iq = self.sdr.read()
            mag = compute_fft(iq)
            freqs = freq_axis(lo)

            stitcher.add(freqs, mag)

        return stitcher.result()

