import matplotlib.pyplot as plt
from pluto import PlutoSDR
from analyzer import SpectrumAnalyzer

CENTER_FREQ = 4.6e8  # Startfrequenz
UPDATE_INTERVAL = 0.1  # Sekunden zwischen Updates

# Pluto und Analyzer initialisieren
sdr = PlutoSDR()
analyzer = SpectrumAnalyzer(sdr)

plt.ion()  # Interaktive Grafik
fig, ax = plt.subplots(figsize=(15,6))
line, = ax.plot([], [])
ax.set_xlabel("Frequenz [MHz]")
ax.set_ylabel("Pegel [dBFS]")
ax.set_title("Kontinuierlicher Wideband-Spektrumanalyzer")
ax.grid(True)

try:
    while True:
        freqs, mags = analyzer.sweep(CENTER_FREQ)

        line.set_data([f/1e6 for f in freqs], mags)
        ax.set_xlim(min(freqs)/1e6, max(freqs)/1e6)
        ax.set_ylim(min(mags)-5, max(mags)+5)
        plt.pause(UPDATE_INTERVAL)

except KeyboardInterrupt:
    print("Beendet")
    plt.ioff()
    plt.show()

