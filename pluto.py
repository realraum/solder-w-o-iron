# pluto.py

import adi
import numpy as np
from config import FS, RX_GAIN_MODE, RX_GAIN

class PlutoSDR:
    def __init__(self, uri="ip:192.168.2.1"):
        self.sdr = adi.Pluto(uri)
        self.sdr.sample_rate = int(FS)
        self.sdr.rx_rf_bandwidth = int(FS)
        self.sdr.rx_gain_control_mode = RX_GAIN_MODE

        if RX_GAIN is not None:
            self.sdr.rx_hardwaregain_chan0 = RX_GAIN

    def tune(self, freq):
        self.sdr.rx_lo = int(freq)

    def read(self):
        self.sdr.rx_buffer_size = 8192  # gleich der FFT
        iq = self.sdr.rx()
        iq -= np.mean(iq)   # DC removal
        return iq

