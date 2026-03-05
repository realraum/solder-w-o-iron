Demo for how to make a spectrum analyzer for the ADALM Pluto w/o knowledge of python, but only fiddling arount with AI.


Python project for a 100 MHz spectrum analyzer, built from 10 MHz steps with 20 MHz tiles (i.e.overlap).

🎛️ Project: Wideband Spectrum Analyzer (100 MHz) 
Key Data Parameter Value 
Total Bandwidth 100 MHz Tile 
Bandwidth 20 MHz 
Step Size 10 MHz 
Overlap 50% FFT 
Size 8192 
Resolution ≈ 2.4 kHz/bin 
Hardware ADALM-Pluto 
Language Python (pyadi-iio)

HowTo

git clone https://github.com/realraum/solder-w-o-iron.git

python -m venv solder-w-o-iron

source solder-w-o-iron/bin/activate

cd solder-w-o-iron

pip install  pyadi-iio

pip install matplotlib

python runcfix.py
