# dsp.py

import numpy as np
from config import FFT_SIZE, WINDOW, FS

def get_window():
    if WINDOW == "blackman":
        return np.blackman(FFT_SIZE)
    elif WINDOW == "hann":
        return np.hanning(FFT_SIZE)
    else:
        return np.ones(FFT_SIZE)

WINDOW_FN = get_window()

def compute_fft(iq):
    iq = iq[:FFT_SIZE] * WINDOW_FN
    spec = np.fft.fftshift(np.fft.fft(iq))
    mag = 20 * np.log10(np.abs(spec) + 1e-12)

    # LO-Spike maskieren
    mid = FFT_SIZE // 2
    mag[mid-3:mid+3] = np.nan

    return mag

def freq_axis(lo):
    return np.linspace(
        lo - FS/2,
        lo + FS/2,
        FFT_SIZE,
        endpoint=False
    )

