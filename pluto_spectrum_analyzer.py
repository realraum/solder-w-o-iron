#!/usr/bin/env python3
"""
ADALM-Pluto Fast Spectrum Analyzer
====================================
Sweept in 20 MHz breiten Segmenten über den gewünschten Frequenzbereich.
DC-Anteile (Mitte jedes Segments) werden ausgeblendet/interpoliert.
Die Segmente werden zu einem vollständigen Spektrum zusammengesetzt.

Abhängigkeiten:
    pip install pyadi-iio numpy scipy matplotlib

Verwendung:
    python pluto_spectrum_analyzer.py
    python pluto_spectrum_analyzer.py --start 70e6 --stop 1000e6 --uri ip:192.168.2.1
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import EngFormatter
from scipy.signal import windows
import adi  # pyadi-iio


# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_URI        = "ip:192.168.2.1"   # PlutoSDR IP-Adresse
DEFAULT_START_HZ   = 400e6               # Startfrequenz Hz
DEFAULT_STOP_HZ    = 600e6             # Stoppfrequenz Hz
SAMPLE_RATE        = 20e6               # Abtastrate = Segmentbreite
FFT_SIZE           = 4096               # FFT-Punkte pro Segment
BUFFER_SAMPLES     = FFT_SIZE * 4       # IQ-Samples pro Erfassung
DC_GUARD_BINS      = 8                  # Bins um DC ausblenden (±)
ATTENUATION_DB     = 0                  # RX Attenuation (0 = max Gain)
RF_BANDWIDTH       = 18_000_000         # Analoge Bandbreite (leicht < 20 MHz)
AVERAGES           = 2                  # FFT-Mittelungen pro Segment
OVERLAP_BINS       = 64                 # Überlappung zum Blending der Segmentkanten


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def compute_fft_segment(sdr, center_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Empfängt IQ-Daten bei center_hz und berechnet ein gemitteltes FFT-Segment.

    Gibt zurück:
        freqs_hz  : Frequenzachse des Segments (Hz)
        psd_db    : Leistungsdichte in dBFS
    """
    sdr.rx_lo = int(center_hz)

    # Fenster für Spektralleck-Unterdrückung
    win = windows.blackmanharris(FFT_SIZE)
    win_cg = np.sum(win) ** 2  # Kohärenter Gewinn

    psd_linear = np.zeros(FFT_SIZE)

    for _ in range(AVERAGES):
        samples = sdr.rx()
        if len(samples) < FFT_SIZE:
            samples = np.pad(samples, (0, FFT_SIZE - len(samples)))
        chunk = samples[:FFT_SIZE].astype(np.complex64)

        spectrum = np.fft.fftshift(np.fft.fft(chunk * win, n=FFT_SIZE))
        psd_linear += (np.abs(spectrum) ** 2) / (win_cg * AVERAGES)

    # dBFS normiert auf FFT-Größe
    psd_db = 10 * np.log10(psd_linear / FFT_SIZE + 1e-20)

    # Frequenzachse
    freqs_hz = center_hz + np.fft.fftshift(
        np.fft.fftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
    )

    # ── DC-Unterdrückung ──────────────────────────────────────────────────────
    # Bins nahe DC werden durch lineare Interpolation der Nachbarbins ersetzt
    mid = FFT_SIZE // 2
    lo  = mid - DC_GUARD_BINS
    hi  = mid + DC_GUARD_BINS + 1

    # Interpoliere über den DC-Bereich
    x_known = np.array([lo - 1, hi])
    y_known = np.array([psd_db[lo - 1], psd_db[hi]])
    psd_db[lo:hi] = np.interp(np.arange(lo, hi), x_known, y_known)

    return freqs_hz, psd_db


def build_frequency_plan(start_hz: float, stop_hz: float) -> list[float]:
    """
    Erstellt eine Liste von Mittenfrequenzen für die Segmente.
    Jedes Segment hat SAMPLE_RATE Breite.
    Die Segmente werden so versetzt, dass Anfang und Ende abgedeckt sind.
    """
    centers = []
    fc = start_hz + SAMPLE_RATE / 2
    while fc - SAMPLE_RATE / 2 < stop_hz:
        centers.append(fc)
        fc += SAMPLE_RATE
    return centers


def blend_segments(all_freqs: list, all_psds: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Setzt Segmente zu einem kontinuierlichen Spektrum zusammen.
    Randbereiche werden cosinus-gewichtet geblendet, um Übergänge zu glätten.
    """
    # Alle Arrays zusammenführen und nach Frequenz sortieren
    freqs = np.concatenate(all_freqs)
    psds  = np.concatenate(all_psds)

    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    psds  = psds[sort_idx]

    # Duplikate entfernen (Überlappbereiche mitteln)
    _, unique_idx, counts = np.unique(
        np.round(freqs / 1e3).astype(int),  # 1-kHz-Raster für Duplikaterkennung
        return_index=True,
        return_counts=True
    )
    freqs_out = freqs[unique_idx]
    psds_out  = psds[unique_idx]

    return freqs_out, psds_out


# ─────────────────────────────────────────────────────────────────────────────
# SDR-Initialisierung
# ─────────────────────────────────────────────────────────────────────────────

def init_sdr(uri: str) -> adi.Pluto:
    print(f"Verbinde mit PlutoSDR: {uri}")
    sdr = adi.Pluto(uri=uri)

    sdr.sample_rate          = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth      = int(RF_BANDWIDTH)
    sdr.rx_buffer_size       = BUFFER_SAMPLES
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0   = -ATTENUATION_DB  # positiv = mehr Gain
    sdr.rx_enabled_channels     = [0]

    print(f"  Abtastrate   : {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"  FFT-Größe    : {FFT_SIZE}")
    print(f"  DC-Guard     : ±{DC_GUARD_BINS} Bins")
    print(f"  Mittelungen  : {AVERAGES}x pro Segment")
    return sdr


# ─────────────────────────────────────────────────────────────────────────────
# Sweep-Funktion
# ─────────────────────────────────────────────────────────────────────────────

def do_sweep(sdr: adi.Pluto, centers: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Führt einen vollständigen Sweep durch und gibt das zusammengesetzte Spektrum zurück."""
    all_freqs = []
    all_psds  = []

    for fc in centers:
        freqs, psd = compute_fft_segment(sdr, fc)
        all_freqs.append(freqs)
        all_psds.append(psd)

    return blend_segments(all_freqs, all_psds)


# ─────────────────────────────────────────────────────────────────────────────
# Live-Plot
# ─────────────────────────────────────────────────────────────────────────────

def run_live(args):
    sdr     = init_sdr(args.uri)
    centers = build_frequency_plan(args.start, args.stop)
    n_segs  = len(centers)
    span_mhz = (args.stop - args.start) / 1e6

    print(f"\nSweep-Plan: {n_segs} Segmente × {SAMPLE_RATE/1e6:.0f} MHz "
          f"= {span_mhz:.0f} MHz Gesamtspan")
    print("Starte Live-Anzeige … (Fenster schließen zum Beenden)\n")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#58a6ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.set_xlim(args.start / 1e6, args.stop / 1e6)
    ax.set_ylim(-120, 0)
    ax.set_xlabel("Frequenz (MHz)")
    ax.set_ylabel("Leistung (dBFS)")
    ax.set_title(f"ADALM-Pluto Spectrum Analyzer  |  {span_mhz:.0f} MHz Span")
    ax.xaxis.set_major_formatter(EngFormatter(unit="", sep=""))
    ax.grid(True, color="#21262d", linewidth=0.6)

    # Segment-Grenzen einzeichnen
    for fc in centers:
        ax.axvline((fc - SAMPLE_RATE / 2) / 1e6,
                   color="#388bfd22", linewidth=0.5, linestyle="--")

    line, = ax.plot([], [], color="#39d353", linewidth=0.8, antialiased=True)
    peak_line = ax.axhline(-120, color="#f78166", linewidth=0.5, linestyle=":")
    peak_text = ax.text(0.01, 0.96, "", transform=ax.transAxes,
                        color="#f78166", fontsize=8, va="top")
    fps_text  = ax.text(0.99, 0.96, "", transform=ax.transAxes,
                        color="#8b949e", fontsize=8, va="top", ha="right")

    hold_psd = None  # Max-Hold Buffer

    def update(_frame):
        nonlocal hold_psd
        t0 = time.perf_counter()

        freqs, psd = do_sweep(sdr, centers)
        freqs_mhz  = freqs / 1e6

        # Max-Hold
        if hold_psd is None or len(hold_psd) != len(psd):
            hold_psd = psd.copy()
        else:
            hold_psd = np.maximum(hold_psd, psd)

        line.set_data(freqs_mhz, psd)
        ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])

        # Peak-Markierung
        peak_idx = np.argmax(psd)
        peak_db  = psd[peak_idx]
        peak_hz  = freqs[peak_idx]
        peak_line.set_ydata([peak_db, peak_db])
        peak_text.set_text(f"Peak: {peak_hz/1e6:.3f} MHz  {peak_db:.1f} dBFS")

        elapsed = time.perf_counter() - t0
        fps_text.set_text(f"Sweep: {elapsed*1000:.0f} ms  ({1/elapsed:.1f} Hz)")

        return line, peak_line, peak_text, fps_text

    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=True, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Einzel-Sweep mit Plot speichern
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args):
    sdr     = init_sdr(args.uri)
    centers = build_frequency_plan(args.start, args.stop)

    print(f"Einzelner Sweep über {len(centers)} Segmente …")
    t0 = time.perf_counter()
    freqs, psd = do_sweep(sdr, centers)
    elapsed = time.perf_counter() - t0
    print(f"Sweep abgeschlossen in {elapsed*1000:.0f} ms")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(freqs / 1e6, psd, linewidth=0.8, color="tab:green")
    ax.set_xlabel("Frequenz (MHz)")
    ax.set_ylabel("Leistung (dBFS)")
    ax.set_title(f"Spektrum  {args.start/1e6:.0f}–{args.stop/1e6:.0f} MHz  "
                 f"({elapsed*1000:.0f} ms Sweep)")
    ax.set_ylim(-120, 0)
    ax.grid(True, alpha=0.4)
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
        description="ADALM-Pluto Spectrum Analyzer mit 20-MHz-Segmentsweep"
    )
    p.add_argument("--uri",   default=DEFAULT_URI,
                   help=f"PlutoSDR URI (default: {DEFAULT_URI})")
    p.add_argument("--start", type=float, default=DEFAULT_START_HZ,
                   help="Startfrequenz in Hz (default: 70e6)")
    p.add_argument("--stop",  type=float, default=DEFAULT_STOP_HZ,
                   help="Stoppfrequenz in Hz (default: 1000e6)")
    p.add_argument("--fft",   type=int,   default=FFT_SIZE,
                   help=f"FFT-Größe (default: {FFT_SIZE})")
    p.add_argument("--avg",   type=int,   default=AVERAGES,
                   help=f"Mittelungen pro Segment (default: {AVERAGES})")
    p.add_argument("--gain",  type=float, default=ATTENUATION_DB,
                   help="RX Attenuation in dB (0 = max Gain, default: 0)")
    p.add_argument("--single", action="store_true",
                   help="Nur einen einzelnen Sweep durchführen")
    p.add_argument("--save",  default=None,
                   help="Speichert den Plot als Bilddatei (nur mit --single)")
    return p.parse_args()


def main():
    args = parse_args()

    # Globale Konstanten aus CLI überschreiben
    global FFT_SIZE, AVERAGES, ATTENUATION_DB
    FFT_SIZE       = args.fft
    AVERAGES       = args.avg
    ATTENUATION_DB = args.gain

    print("=" * 55)
    print("  ADALM-Pluto Spectrum Analyzer")
    print("=" * 55)
    print(f"  Bereich : {args.start/1e6:.1f} – {args.stop/1e6:.1f} MHz")
    print(f"  Segment : {SAMPLE_RATE/1e6:.0f} MHz breit")
    print(f"  FFT     : {FFT_SIZE} Punkte  →  {SAMPLE_RATE/FFT_SIZE/1e3:.1f} kHz/Bin")
    print(f"  DC-Guard: ±{DC_GUARD_BINS} Bins = ±{DC_GUARD_BINS*SAMPLE_RATE/FFT_SIZE/1e3:.0f} kHz")
    print("=" * 55)

    if args.single:
        run_single(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
