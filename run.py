# run.py

from pluto import PlutoSDR
from analyzer import SpectrumAnalyzer
import matplotlib.pyplot as plt

#CENTER_FREQ = 2.4e9  # Beispiel: 2.4 GHz ISM
CENTER_FREQ = 1.2e8  # Beispiel: 2.4 GHz ISM

sdr = PlutoSDR()
analyzer = SpectrumAnalyzer(sdr)

freqs, mags = analyzer.sweep(CENTER_FREQ)

plt.figure(figsize=(15,6))
plt.plot([f/1e6 for f in freqs], mags)
plt.xlabel("Frequenz [MHz]")
plt.ylabel("Pegel [dBFS]")
plt.title("100 MHz Wideband-Spektrumanalyzer (ADALM-Pluto)")
plt.grid(True)
plt.show()

