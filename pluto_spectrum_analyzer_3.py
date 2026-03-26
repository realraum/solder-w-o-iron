#!/usr/bin/env python3
"""
ADALM-Pluto Fast Spectrum Analyzer  –  Seitenband-Sweep
=========================================================
Strategie:
  - Abtastrate = maximal (61.44 MSPS) für größtmögliche Nyquist-Bandbreite
  - Pro Segment wird nur das POSITIVE Seitenband (USB) benutzt:
      von  LO + DC_GUARD_HZ  bis  LO + SAMPLE_RATE/2
  - Der LO springt bei jedem Schritt um (SAMPLE_RATE/2 - DC_GUARD_HZ)
  - DC und das negative Seitenband werden komplett verworfen
  - Typisch nutzbare Breite pro Segment ≈ 28.7 MHz (bei 61.44 MSPS, 2 MHz DC-Guard)

Vorteil gegenüber beidseitigem Sweep:
  ✓ Keine DC-Artefakte im Nutzspektrum
  ✓ Maximale Sweep-Geschwindigkeit durch weniger benötigte Segmente
  ✓ Sauberere Segmentgrenzen (kein Überlapp nötig)

Abhängigkeiten:
    pip install pyadi-iio numpy scipy matplotlib

Verwendung:
    python pluto_spectrum_analyzer.py
    python pluto_spectrum_analyzer.py --start 70e6 --stop 1000e6
    python pluto_spectrum_analyzer.py --rate 56e6 --guard 2e6
    python pluto_spectrum_analyzer.py --single --save spektrum.png
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import EngFormatter, MultipleLocator
from scipy.signal import windows
import adi  # pyadi-iio


# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_URI       = "ip:192.168.2.1"
DEFAULT_START_HZ  = 70e6
DEFAULT_STOP_HZ   = 1000e6

# Pluto max: 61.44 MSPS  (geräteseitig auf gültige Werte gerundet)
# Empfehlung: 61.44e6, 56e6, 40e6, 20e6
SAMPLE_RATE       = 61.44e6        # MSPS – so hoch wie möglich

# DC-Guard: untere Grenze des genutzten Seitenbands ab LO
# Pluto hat typisch ~100–500 kHz DC-Spike; 1–3 MHz Puffer ist sicher
DC_GUARD_HZ       = 2e6            # Hz unterhalb derer wir nicht nutzen

FFT_SIZE          = 8192           # Punkte – groß für hohe Auflösung bei hoher Rate
BUFFER_SAMPLES    = FFT_SIZE * 2   # IQ-Samples im Puffer
AVERAGES          = 1              # Mittelungen pro Segment (1 = schnellstmöglich)
RX_GAIN_DB        = 30             # Hardware-Gain in dB (0–73)

# Analoger HF-Filter des AD9363: 200 kHz – 56 MHz
RF_BANDWIDTH      = 56_000_000     # Hz


# ─────────────────────────────────────────────────────────────────────────────
# Sweep-Planung
# ─────────────────────────────────────────────────────────────────────────────

def sideband_plan(start_hz: float, stop_hz: float,
                  fs: float, dc_guard: float) -> list[dict]:
    """
    Berechnet LO-Positionen für den Seitenband-Sweep.

    Jedes Segment nutzt nur:
        f_low  = LO + dc_guard
        f_high = LO + fs/2

    Nutzbandbreite pro Segment:
        bw_use = fs/2 - dc_guard

    Der LO startet so, dass f_low == start_hz, dann springt er
    um bw_use weiter bis stop_hz abgedeckt ist.

    Gibt eine Liste von Dicts zurück:
        lo        : LO-Frequenz in Hz
        f_low     : unterste genutzte Frequenz
        f_high    : oberste genutzte Frequenz
        bin_start : erster FFT-Bin (nach fftshift), der genutzt wird
        bin_stop  : letzter FFT-Bin (exklusiv)
    """
    # Pluto AD9364/AD9363 LO-Grenzen
    LO_MIN = 70e6
    LO_MAX = 6000e6

    bw_use     = fs / 2.0 - dc_guard      # z.B. 30.72 MHz - 2 MHz = 28.72 MHz
    segments   = []
    guard_bins = int(np.ceil(dc_guard / (fs / FFT_SIZE)))

    lo_ideal = start_hz - dc_guard        # erstes LO so, dass f_low = start_hz

    # Falls lo_ideal < LO_MIN: LO auf LO_MIN klemmen und
    # die tatsächlich genutzte Unterfrequenz anpassen.
    lo = max(lo_ideal, LO_MIN)

    while True:
        lo = min(lo, LO_MAX)              # obere Grenze ebenfalls sichern
        f_low  = lo + dc_guard
        f_high = lo + fs / 2.0

        # Wenn durch LO-Klemmen f_low > stop_hz: fertig
        if f_low > stop_hz:
            break

        # Bin-Grenzen für dieses (ggf. geklemmte) LO neu berechnen
        actual_guard_bins = int(np.ceil(
            max(dc_guard, f_low - lo) / (fs / FFT_SIZE)
        ))

        segments.append(dict(
            lo        = lo,
            f_low     = f_low,
            f_high    = min(f_high, stop_hz),
            bin_start = FFT_SIZE // 2 + actual_guard_bins,
            bin_stop  = FFT_SIZE,
        ))

        if f_high >= stop_hz:
            break
        lo += bw_use

    if not segments:
        raise ValueError(
            f"Kein gültiges Segment gefunden. "
            f"Start {start_hz/1e6:.1f} MHz liegt möglicherweise "
            f"unter Pluto-Minimum (70 MHz + DC-Guard)."
        )
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# FFT-Berechnung
# ─────────────────────────────────────────────────────────────────────────────

_window_cache: dict = {}

def get_window(n: int) -> tuple[np.ndarray, float]:
    if n not in _window_cache:
        w = windows.blackmanharris(n).astype(np.float32)
        _window_cache[n] = (w, float(np.sum(w) ** 2))
    return _window_cache[n]


def compute_segment(sdr, seg: dict, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Stellt den LO ein, erfasst IQ-Daten, berechnet FFT und
    gibt nur das positive Seitenband ab DC_GUARD zurück.

    Gibt zurück:
        freqs_hz : absolute Frequenzachse der genutzten Bins
        psd_db   : Leistung in dBFS
    """
    sdr.rx_lo = int(seg["lo"])

    win, win_cg = get_window(FFT_SIZE)
    psd_lin = np.zeros(FFT_SIZE, dtype=np.float64)

    for _ in range(AVERAGES):
        raw = sdr.rx()
        if len(raw) < FFT_SIZE:
            raw = np.pad(raw, (0, FFT_SIZE - len(raw)))
        chunk = raw[:FFT_SIZE].astype(np.complex64)

        spec = np.fft.fftshift(np.fft.fft(chunk * win, n=FFT_SIZE))
        psd_lin += np.abs(spec).astype(np.float64) ** 2

    psd_lin /= (win_cg * AVERAGES)
    psd_db   = 10.0 * np.log10(psd_lin / FFT_SIZE + 1e-20)

    # Nur positives Seitenband ab DC-Guard verwenden
    bs = seg["bin_start"]
    be = seg["bin_stop"]
    used_psd = psd_db[bs:be]

    # Absolute Frequenzachse für die genutzten Bins
    bin_freqs = seg["lo"] + np.arange(bs, be) * (fs / FFT_SIZE) - fs / 2.0

    # Letztes Segment: Frequenzen über stop_hz abschneiden
    mask = bin_freqs <= seg["f_high"] + fs / FFT_SIZE
    return bin_freqs[mask], used_psd[mask]


# ─────────────────────────────────────────────────────────────────────────────
# SDR-Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

def init_sdr(uri: str, fs: float) -> adi.Pluto:
    print(f"Verbinde mit PlutoSDR: {uri}")
    sdr = adi.Pluto(uri=uri)

    sdr.sample_rate             = int(fs)
    sdr.rx_rf_bandwidth         = int(RF_BANDWIDTH)
    sdr.rx_buffer_size          = BUFFER_SAMPLES
    sdr.gain_control_mode_chan0  = "manual"
    sdr.rx_hardwaregain_chan0    = RX_GAIN_DB
    sdr.rx_enabled_channels     = [0]

    actual_fs = sdr.sample_rate
    print(f"  Abtastrate (gesetzt)  : {fs/1e6:.3f} MSPS")
    print(f"  Abtastrate (Hardware) : {actual_fs/1e6:.3f} MSPS")
    print(f"  HF-Bandbreite         : {RF_BANDWIDTH/1e6:.0f} MHz")
    print(f"  RX Gain               : {RX_GAIN_DB} dB")
    print(f"  FFT-Größe             : {FFT_SIZE}")
    return sdr


# ─────────────────────────────────────────────────────────────────────────────
# Vollständiger Sweep
# ─────────────────────────────────────────────────────────────────────────────

def do_sweep(sdr, segments: list, fs: float) -> tuple[np.ndarray, np.ndarray]:
    all_f, all_p = [], []
    for seg in segments:
        f, p = compute_segment(sdr, seg, fs)
        all_f.append(f)
        all_p.append(p)
    freqs = np.concatenate(all_f)
    psds  = np.concatenate(all_p)
    order = np.argsort(freqs)
    return freqs[order], psds[order]


# ─────────────────────────────────────────────────────────────────────────────
# Live-Plot
# ─────────────────────────────────────────────────────────────────────────────

def run_live(args, sdr, segments, fs):
    n_segs   = len(segments)
    bw_use   = fs / 2 - args.guard
    span_mhz = (args.stop - args.start) / 1e6

    print(f"\nSweep-Plan:")
    print(f"  Samplerate      : {fs/1e6:.3f} MSPS")
    print(f"  DC-Guard        : {args.guard/1e6:.1f} MHz")
    print(f"  Nutzband/Segment: {bw_use/1e6:.2f} MHz")
    print(f"  Segmente        : {n_segs}")
    print(f"  Gesamtspan      : {span_mhz:.0f} MHz")
    print(f"  Auflösung       : {fs/FFT_SIZE/1e3:.2f} kHz/Bin\n")

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor("#080c10")
    ax.set_facecolor("#080c10")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2a38")
    ax.tick_params(colors="#7a8fa8", labelsize=8)
    ax.xaxis.label.set_color("#7a8fa8")
    ax.yaxis.label.set_color("#7a8fa8")
    ax.title.set_color("#4db8ff")

    ax.set_xlim(args.start / 1e6, args.stop / 1e6)
    ax.set_ylim(-110, 10)
    ax.set_xlabel("Frequenz (MHz)")
    ax.set_ylabel("Leistung (dBFS)")
    ax.set_title(
        f"ADALM-Pluto Spectrum Analyzer  ·  "
        f"{args.start/1e6:.0f}–{args.stop/1e6:.0f} MHz  ·  "
        f"{fs/1e6:.2f} MSPS  ·  {bw_use/1e6:.1f} MHz/Seg"
    )
    ax.xaxis.set_major_formatter(EngFormatter(unit="", sep=""))
    ax.grid(True, color="#0e1820", linewidth=0.7)
    ax.grid(True, which="minor", color="#0b1318", linewidth=0.3)
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    # Segment-Grenzen einzeichnen
    for seg in segments:
        ax.axvline(seg["f_low"] / 1e6,
                   color="#1a3a5c", linewidth=0.6, linestyle=":")

    line_live, = ax.plot([], [], color="#00e5ff", linewidth=0.7,
                         alpha=0.9, label="Live")
    line_hold, = ax.plot([], [], color="#ff6b35", linewidth=0.5,
                         alpha=0.6, label="Max Hold")
    ax.legend(loc="upper right", fontsize=7,
              facecolor="#0d1a26", edgecolor="#1e2a38", labelcolor="#7a8fa8")

    info_peak = ax.text(0.01, 0.97, "", transform=ax.transAxes,
                        color="#ff6b35", fontsize=8, va="top",
                        fontfamily="monospace")
    info_fps  = ax.text(0.99, 0.97, "", transform=ax.transAxes,
                        color="#4db8ff", fontsize=8, va="top", ha="right",
                        fontfamily="monospace")

    hold_psd = [None]

    def update(_frame):
        t0 = time.perf_counter()
        freqs, psd = do_sweep(sdr, segments, fs)
        elapsed = time.perf_counter() - t0

        fmhz = freqs / 1e6
        line_live.set_data(fmhz, psd)

        if hold_psd[0] is None or len(hold_psd[0]) != len(psd):
            hold_psd[0] = psd.copy()
        else:
            hold_psd[0] = np.maximum(hold_psd[0], psd)
        line_hold.set_data(fmhz, hold_psd[0])

        pidx = int(np.argmax(psd))
        info_peak.set_text(
            f"Peak  {freqs[pidx]/1e6:.4f} MHz  {psd[pidx]:.1f} dBFS"
        )
        info_fps.set_text(
            f"{elapsed*1e3:.0f} ms/sweep  {1/elapsed:.1f} Hz  {n_segs} segs"
        )
        return line_live, line_hold, info_peak, info_fps

    def init_plot():
        line_live.set_data([], [])
        line_hold.set_data([], [])
        info_peak.set_text("")
        info_fps.set_text("")
        return line_live, line_hold, info_peak, info_fps

    ani = animation.FuncAnimation(
        fig, update, init_func=init_plot,
        interval=1, blit=True, cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Einzel-Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args, sdr, segments, fs):
    n = len(segments)
    print(f"Einzelsweep: {n} Segmente …")
    t0 = time.perf_counter()
    freqs, psd = do_sweep(sdr, segments, fs)
    elapsed = time.perf_counter() - t0
    print(f"Fertig in {elapsed*1e3:.0f} ms  "
          f"({len(freqs)} Punkte, {fs/FFT_SIZE/1e3:.2f} kHz/Bin)")

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(freqs / 1e6, psd, linewidth=0.7, color="#00e5ff")
    ax.set_xlabel("Frequenz (MHz)")
    ax.set_ylabel("Leistung (dBFS)")
    ax.set_title(
        f"Spektrum  {args.start/1e6:.0f}–{args.stop/1e6:.0f} MHz  "
        f"({elapsed*1e3:.0f} ms,  {fs/1e6:.2f} MSPS,  {n} Segmente)"
    )
    ax.set_ylim(-110, 10)
    ax.xaxis.set_major_formatter(EngFormatter(unit="", sep=""))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Gespeichert: {args.save}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ADALM-Pluto Seitenband Spectrum Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uri",    default=DEFAULT_URI,   help="PlutoSDR URI")
    p.add_argument("--start",  type=float, default=DEFAULT_START_HZ,  help="Startfrequenz Hz")
    p.add_argument("--stop",   type=float, default=DEFAULT_STOP_HZ,   help="Stoppfrequenz Hz")
    p.add_argument("--rate",   type=float, default=SAMPLE_RATE,       help="Abtastrate Hz (max 61.44e6)")
    p.add_argument("--guard",  type=float, default=DC_GUARD_HZ,       help="DC-Guard Hz")
    p.add_argument("--fft",    type=int,   default=FFT_SIZE,          help="FFT-Größe")
    p.add_argument("--avg",    type=int,   default=AVERAGES,          help="Mittelungen/Segment")
    p.add_argument("--gain",   type=float, default=RX_GAIN_DB,        help="RX Gain dB (0–73)")
    p.add_argument("--single", action="store_true",                   help="Einzelsweep")
    p.add_argument("--save",   default=None,                          help="Plot speichern (PNG/SVG)")
    return p.parse_args()


def main():
    args = parse_args()

    global FFT_SIZE, AVERAGES, RX_GAIN_DB
    FFT_SIZE   = args.fft
    AVERAGES   = args.avg
    RX_GAIN_DB = args.gain

    fs     = args.rate
    bw_use = fs / 2 - args.guard

    print("=" * 60)
    print("  ADALM-Pluto Seitenband Spectrum Analyzer")
    print("=" * 60)
    print(f"  Bereich    : {args.start/1e6:.1f} – {args.stop/1e6:.1f} MHz")
    print(f"  Samplerate : {fs/1e6:.3f} MSPS")
    print(f"  DC-Guard   : {args.guard/1e6:.2f} MHz")
    print(f"  Nutzband   : {args.guard/1e6:.2f} .. {fs/2/1e6:.3f} MHz ab LO  ({bw_use/1e6:.3f} MHz)")
    print(f"  FFT        : {FFT_SIZE} Punkte  →  {fs/FFT_SIZE/1e3:.2f} kHz/Bin")

    segments = sideband_plan(args.start, args.stop, fs, args.guard)
    n = len(segments)
    print(f"  Segmente   : {n}")
    print("=" * 60)

    sdr = init_sdr(args.uri, fs)

    if args.single:
        run_single(args, sdr, segments, fs)
    else:
        run_live(args, sdr, segments, fs)


if __name__ == "__main__":
    main()
