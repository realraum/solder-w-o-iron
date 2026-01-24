import matplotlib.pyplot as plt
from pluto import PlutoSDR
from analyzer import SpectrumAnalyzer

#CENTER_FREQ = 2.4e9      # Startfrequenz
CENTER_FREQ = 2.8e9      # Startfrequenz
UPDATE_INTERVAL = 0.6     # Sekunden zwischen Updates
Y_MIN = 60              # feste untere Grenze dBFS
Y_MAX = 110                 # feste obere Grenze dBFS

# Pluto und Analyzer initialisieren
sdr = PlutoSDR()
analyzer = SpectrumAnalyzer(sdr)

plt.ion()  # Interaktive Grafik
fig, ax = plt.subplots(figsize=(15,6))
line, = ax.plot([], [])
ax.set_xlabel("Frequenz [MHz]")
ax.set_ylabel("Pegel [dBFS]")
ax.set_title("Kontinuierlicher Wideband-Spektrumanalysator")
ax.set_ylim(Y_MIN, Y_MAX)  # feste Y-Skalierung
ax.grid(True)

try:
    while True:
        freqs, mags = analyzer.sweep(CENTER_FREQ)

        line.set_data([f/1e6 for f in freqs], mags)
        ax.set_xlim(min(freqs)/1e6, max(freqs)/1e6)  # X-Skala dynamisch
        plt.pause(UPDATE_INTERVAL)

except KeyboardInterrupt:
    print("Beendet")
    plt.ioff()
    plt.show()

