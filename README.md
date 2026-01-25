Python-Projekt für einen 100-MHz-Spektrumanalyzer, aufgebaut aus 10-MHz-Schritten mit 20-MHz-Kacheln (also sauberer Überlappung).


🎛️ Projekt: Wideband-Spektrumanalyzer (100 MHz)
Eckdaten
Parameter	Wert
Gesamtbandbreite	100 MHz
Kachelbandbreite	20 MHz
Schrittweite	10 MHz
Überlappung	50 %
FFT-Größe	8192
Auflösung	≈ 2.4 kHz/bin
Hardware	ADALM-Pluto
Sprache	Python (pyadi-iio)

git clone https://github.com/realraum/solder-w-o-iron.git

python -m venv solder-w-o-iron

source solder-w-o-iron/bin/activate
cd solder-w-o-iron

pip install  pyadi-iio
pip install matplotlib

python runcfix.py
