# stitcher.py

class SpectrumStitcher:
    def __init__(self):
        self.sum = {}
        self.count = {}

    def add(self, freqs, mags):
        for f, m in zip(freqs, mags):
            f = round(f, 1)
            if f not in self.sum:
                self.sum[f] = m
                self.count[f] = 1
            else:
                self.sum[f] += m
                self.count[f] += 1

    def result(self):
        freqs = sorted(self.sum.keys())
        mags = [self.sum[f] / self.count[f] for f in freqs]
        return freqs, mags

