#!/usr/bin/env python3
"""
pluto_sigmf_recorder_2.py
========================
Nimmt von einem ADALM Pluto (Rev. C/D) beide RX-Kanaele gleichzeitig I/Q-Daten auf
und speichert sie in konfigurierbaren Bloecken im SigMF-Format (.sigmf-data + .sigmf-meta).

Zweistufige Kalibrierung:
    Stufe 1 – Interner Loopback (automatisch):
        Nutzt den AD9361-internen TX->RX-Pfad. Misst den Phasenoffset
        der digitalen Signalkette inkl. Chip-Analogpfad bis zum Balun.

    Stufe 2 – Externes Referenzsignal (interaktiv):
        Das Skript pausiert und fordert den Benutzer auf, ein bekanntes
        Referenzsignal auf beide Antennen zu geben (z.B. Splitter von
        Signalgenerator, oder Quelle in definiertem Winkel=0). Misst
        den Gesamt-Offset inkl. Kabel und Antennen.

        Δφ_kabel = Δφ_gesamt - Δφ_chip   (wird separat gespeichert)

    Der finale Korrekturfaktor basiert auf Δφ_gesamt und wird auf alle
    RX1-Samples angewendet.

Ablauf:
    1. Pluto konfigurieren
    2. Stufe 1: TX-Loopback, Chip-Offset messen
    3. Stufe 2: Benutzer-Prompt, externes Signal, Gesamt-Offset messen
    4. Aufnahmeschleife mit Phasenkorrektur auf RX1

Abhaengigkeiten:
    pip install pyadi-iio numpy

Verwendung:
    python pluto_sigmf_recorder.py --freq-mhz 433.9 --samplerate-msps 4
    python pluto_sigmf_recorder.py --skip-ext-cal     # nur Stufe 1
    python pluto_sigmf_recorder.py --no-correction    # Offsets nur loggen
    python pluto_sigmf_recorder.py --cal-buffers 16   # mehr Mittelung
"""

import adi
import numpy as np
import argparse
import json
import time
import os
import signal
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Standardwerte
# ---------------------------------------------------------------------------
DEFAULT_URI          = "ip:192.168.2.1"
DEFAULT_FREQ_MHZ     = 433.9
DEFAULT_BW_MHZ       = 4.0
DEFAULT_SR_MSPS      = 4.0
DEFAULT_BLOCK_MIN    = 0.01
DEFAULT_GAIN_DB      = 40
DEFAULT_OUTDIR       = "pluto_recordings"
DEFAULT_BUFFER_SIZE  = 2**18
DEFAULT_CAL_BUFFERS  = 8
DEFAULT_TX_GAIN_DB   = -10
CAL_TONE_OFFSET_HZ   = 100_000   # Kalibrierungston: LO + 100 kHz

# ---------------------------------------------------------------------------
# SigMF / Datei-Hilfsfunktionen
# ---------------------------------------------------------------------------

def samples_for_duration(samplerate: float, seconds: float) -> int:
    return int(samplerate * seconds)


def build_basename(outdir: str, freq_hz: int, sr_sps: int, channel: int,
                   block_index: int, ts: datetime) -> Path:
    ts_str = ts.strftime("%Y%m%d_%H%M%SZ")
    fname  = f"{ts_str}_{freq_hz}Hz_{sr_sps}sps_RX{channel}_block{block_index:04d}"
    return Path(outdir) / fname


def write_sigmf_meta(base: Path, freq_hz: int, sr_sps: int, gain_db: int,
                     ts: datetime, cal: dict) -> None:
    """
    Schreibt SigMF 1.0-Metadaten inkl. vollstaendiger Kalibrierungsinformation.

    Felder:
      interferometry:cal_chip_phase_deg     Offset aus internem Loopback (Stufe 1)
      interferometry:cal_total_phase_deg    Offset aus externem Signal   (Stufe 2)
      interferometry:cal_cable_phase_deg    Differenz = Kabelanteil
      interferometry:correction_applied     ob Korrektur auf RX1 angewendet wurde
      interferometry:correction_source      "external" oder "chip_only"
    """
    meta = {
        "global": {
            "core:datatype":    "cf32_le",
            "core:sample_rate": sr_sps,
            "core:version":     "1.0.0",
            "core:hw":          "ADALM-Pluto Rev.C/D (AD9361)",
            "core:recorder":    "pluto_sigmf_recorder.py",
            "core:description": (
                f"Dual-RX Interferometer-Aufnahme, {freq_hz / 1e6:.6f} MHz, "
                f"{sr_sps / 1e6:.3f} MSPS, Gain {gain_db} dB"
            ),
            # --- Stufe 1: Chip-interner Offset ---
            "interferometry:cal_chip_phase_rad":       cal.get("chip_phase_rad"),
            "interferometry:cal_chip_phase_deg":       cal.get("chip_phase_deg"),
            "interferometry:cal_chip_amplitude_ratio": cal.get("chip_amplitude_ratio"),
            "interferometry:cal_chip_snr_db":          cal.get("chip_snr_db"),
            "interferometry:cal_chip_timestamp":       cal.get("chip_timestamp"),
            # --- Stufe 2: Gesamt-Offset (extern) ---
            "interferometry:cal_total_phase_rad":      cal.get("total_phase_rad"),
            "interferometry:cal_total_phase_deg":      cal.get("total_phase_deg"),
            "interferometry:cal_total_amplitude_ratio":cal.get("total_amplitude_ratio"),
            "interferometry:cal_total_snr_db":         cal.get("total_snr_db"),
            "interferometry:cal_total_timestamp":      cal.get("total_timestamp"),
            # --- Abgeleiteter Kabelanteil ---
            "interferometry:cal_cable_phase_rad":      cal.get("cable_phase_rad"),
            "interferometry:cal_cable_phase_deg":      cal.get("cable_phase_deg"),
            # --- Korrekturstatus ---
            "interferometry:correction_applied":       cal.get("correction_applied", False),
            "interferometry:correction_source":        cal.get("correction_source", "none"),
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency":    freq_hz,
                "core:datetime":     ts.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
                "core:gain":         gain_db,
            }
        ],
        "annotations": [],
    }
    with open(base.with_suffix(".sigmf-meta"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_sigmf_block(outdir: str, freq_hz: int, sr_sps: int, channel: int,
                     block_index: int, ts: datetime, gain_db: int,
                     iq_complex: np.ndarray, cal: dict) -> Path:
    base = build_basename(outdir, freq_hz, sr_sps, channel, block_index, ts)

    iq = np.empty(len(iq_complex) * 2, dtype=np.float32)
    iq[0::2] = iq_complex.real
    iq[1::2] = iq_complex.imag

    data_path = base.with_suffix(".sigmf-data")
    iq.tofile(data_path)
    write_sigmf_meta(base, freq_hz, sr_sps, gain_db, ts, cal)
    return data_path


# ---------------------------------------------------------------------------
# Phasenmessung (gemeinsame Funktion fuer beide Kalibrierungsstufen)
# ---------------------------------------------------------------------------

def measure_phase_offset(rx0: np.ndarray, rx1: np.ndarray,
                          tone_offset_hz: float, sr_sps: int) -> dict:
    """
    Misst Phasendifferenz RX1-RX0 und Amplitudenverhaeltnis am bekannten Ton.
    Methode: FFT-Peak-Extraktion mit Blackman-Fenster.
    """
    n      = len(rx0)
    window = np.blackman(n)
    f0     = np.fft.fft(rx0 * window)
    f1     = np.fft.fft(rx1 * window)

    bin_center   = int(round(tone_offset_hz / sr_sps * n)) % n
    search_width = max(4, int(n * 0.001))
    lo = max(0,   bin_center - search_width)
    hi = min(n-1, bin_center + search_width)
    peak_bin = lo + int(np.argmax(np.abs(f0[lo:hi+1])))

    c0 = f0[peak_bin]
    c1 = f1[peak_bin]

    phase_rad       = float(np.angle(c1 / c0))
    amplitude_ratio = float(np.abs(c1) / (np.abs(c0) + 1e-12))

    mask        = np.ones(n, dtype=bool)
    mask[max(0, peak_bin - search_width):peak_bin + search_width + 1] = False
    noise_floor = np.mean(np.abs(f0[mask]) ** 2)
    snr_db      = float(10 * np.log10(np.abs(c0) ** 2 / (noise_floor + 1e-30)))

    return {
        "phase_rad":       phase_rad,
        "phase_deg":       float(np.degrees(phase_rad)),
        "amplitude_ratio": amplitude_ratio,
        "snr_db":          snr_db,
    }


def collect_measurements(sdr: adi.ad9361, n_buffers: int,
                          tone_offset_hz: float, sr_sps: int,
                          label: str) -> dict:
    """
    Liest n_buffers Puffer, misst jeweils Phasenoffset, gibt gemittelte
    Ergebnisse zurueck. Gibt Warnung bei niedrigem SNR oder hoher Streuung.
    """
    offsets, amplitudes, snrs = [], [], []

    for i in range(n_buffers):
        rx_data = sdr.rx()
        if not isinstance(rx_data, (list, tuple)) or len(rx_data) < 2:
            raise RuntimeError("Nur ein RX-Kanal verfuegbar – Dual-Channel pruefen.")
        rx0 = np.array(rx_data[0], dtype=np.complex64)
        rx1 = np.array(rx_data[1], dtype=np.complex64)
        m   = measure_phase_offset(rx0, rx1, tone_offset_hz, sr_sps)
        offsets.append(m["phase_rad"])
        amplitudes.append(m["amplitude_ratio"])
        snrs.append(m["snr_db"])
        print(f"[CAL ] {label} Puffer {i+1:2d}/{n_buffers}: "
              f"Δφ = {m['phase_deg']:+7.3f}°  "
              f"A1/A0 = {m['amplitude_ratio']:.4f}  "
              f"SNR = {m['snr_db']:.1f} dB")

    mean_phase = float(np.angle(np.mean(np.exp(1j * np.array(offsets)))))
    std_deg    = float(np.degrees(np.std(offsets)))
    mean_amp   = float(np.mean(amplitudes))
    mean_snr   = float(np.mean(snrs))

    print(f"[CAL ] {label} Ergebnis: "
          f"Δφ = {np.degrees(mean_phase):+.4f}°  "
          f"(±{std_deg:.3f}°)  "
          f"A1/A0 = {mean_amp:.4f}  "
          f"SNR = {mean_snr:.1f} dB")

    if mean_snr < 10:
        print(f"[CAL ] *** WARNUNG: SNR {mean_snr:.1f} dB zu niedrig! "
              f"Signal staerker machen oder Gain erhoehen. ***")
    if std_deg > 2.0:
        print(f"[CAL ] *** WARNUNG: Phasenstreuung {std_deg:.2f}° hoch – "
              f"mehr Puffer erwaegen (--cal-buffers). ***")

    return {
        "phase_rad":       mean_phase,
        "phase_deg":       float(np.degrees(mean_phase)),
        "amplitude_ratio": mean_amp,
        "snr_db":          mean_snr,
        "phase_std_deg":   std_deg,
        "timestamp":       datetime.now(tz=timezone.utc).strftime(
                               "%Y-%m-%dT%H:%M:%S.%f") + "Z",
    }


# ---------------------------------------------------------------------------
# Stufe 1: Interner Loopback
# ---------------------------------------------------------------------------

def calibrate_chip(sdr: adi.ad9361, sr_sps: int, freq_hz: int,
                   cal_buffers: int, tx_gain_db: int) -> dict:
    """
    Stufe 1: Misst Chip-internen Phasenoffset via TX->RX-Loopback.
    Konfiguriert TX, sendet CW-Ton, misst, schaltet TX ab.
    """
    print("\n[CAL ] ── Stufe 1: Interner Chip-Loopback ──────────────────────────")

    try:
        sdr.tx_enabled_channels  = [0]
        sdr.tx_lo                = freq_hz
        sdr.tx_rf_bandwidth      = int(sr_sps)
        sdr.tx_hardwaregain_chan0 = tx_gain_db
    except Exception as e:
        print(f"[CAL ] Warnung TX-Setup: {e}")

    t    = np.arange(sdr.rx_buffer_size) / sr_sps
    tone = (np.exp(2j * np.pi * CAL_TONE_OFFSET_HZ * t) * 2**14).astype(np.complex64)

    try:
        sdr.tx_cyclic_buffer = True
        sdr.tx(tone)
        print(f"[CAL ] TX-Ton: {(freq_hz + CAL_TONE_OFFSET_HZ) / 1e6:.6f} MHz  "
              f"(+{CAL_TONE_OFFSET_HZ/1e3:.0f} kHz),  TX-Gain: {tx_gain_db} dB")
    except Exception as e:
        print(f"[CAL ] TX-Ausgabe fehlgeschlagen: {e}")
        _stop_tx(sdr)
        return {}

    time.sleep(0.3)
    for _ in range(3):   # Einschwingen
        sdr.rx()

    result = collect_measurements(sdr, cal_buffers, CAL_TONE_OFFSET_HZ, sr_sps,
                                  label="[S1]")
    _stop_tx(sdr)
    time.sleep(0.2)
    return result


# ---------------------------------------------------------------------------
# Stufe 2: Externes Referenzsignal
# ---------------------------------------------------------------------------

def calibrate_external(sdr: adi.ad9361, sr_sps: int, freq_hz: int,
                        cal_buffers: int) -> dict:
    """
    Stufe 2: Misst Gesamt-Phasenoffset mit externem Referenzsignal.

    Anforderungen an das Referenzsignal:
      - Einzelton bei (freq_hz + CAL_TONE_OFFSET_HZ), also LO + 100 kHz
      - Gleicher Pegel an beiden Antennen (z.B. via Splitter)
        ODER bekannte Quelle in Fernfeld bei Einfallswinkel = 0°
      - Ausreichend stark: SNR > 20 dB empfohlen
    """
    print("\n[CAL ] ── Stufe 2: Externes Referenzsignal ─────────────────────────")
    print(f"[CAL ]")
    print(f"[CAL ]   Bitte jetzt ein Referenzsignal anlegen:")
    print(f"[CAL ]")
    print(f"[CAL ]   Frequenz : {(freq_hz + CAL_TONE_OFFSET_HZ) / 1e6:.6f} MHz")
    print(f"[CAL ]             (= LO + {CAL_TONE_OFFSET_HZ / 1e3:.0f} kHz)")
    print(f"[CAL ]")
    print(f"[CAL ]   Optionen:")
    print(f"[CAL ]   A) Signalgenerator -> Splitter -> beide Antennen")
    print(f"[CAL ]      (kompensiert Kabel + Antennen vollstaendig)")
    print(f"[CAL ]   B) Signalquelle im Fernfeld, Einfallswinkel = 0 deg")
    print(f"[CAL ]      (d = Antennenabstand, Quelle >> d^2/lambda entfernt)")
    print(f"[CAL ]")

    input("[CAL ]   >>> Signal bereit? Enter druecken um zu messen ... ")
    print()

    # Einige Puffer verwerfen (AGC/Filter-Einschwingen nach Signal-Anlegen)
    print(f"[CAL ] Einschwingen (4 Puffer verworfen) ...")
    for _ in range(4):
        sdr.rx()

    result = collect_measurements(sdr, cal_buffers, CAL_TONE_OFFSET_HZ, sr_sps,
                                  label="[S2]")

    input("\n[CAL ]   >>> Referenzsignal entfernen, dann Enter druecken ... ")
    print()
    return result


# ---------------------------------------------------------------------------
# Kalibrierung zusammenfuehren
# ---------------------------------------------------------------------------

def calibrate(sdr: adi.ad9361, sr_sps: int, freq_hz: int,
              cal_buffers: int, tx_gain_db: int,
              skip_ext_cal: bool) -> dict:
    """
    Fuehrt beide Kalibrierungsstufen durch und berechnet den Kabelanteil.
    Gibt ein einheitliches cal-Dict zurueck.
    """
    # --- Stufe 1 ---
    chip = calibrate_chip(sdr, sr_sps, freq_hz, cal_buffers, tx_gain_db)

    # --- Stufe 2 ---
    if skip_ext_cal:
        print("\n[CAL ] Stufe 2 uebersprungen (--skip-ext-cal).")
        print("[CAL ] Korrektur basiert nur auf Chip-Offset – Kabel NICHT kompensiert!")
        ext = {}
    else:
        ext = calibrate_external(sdr, sr_sps, freq_hz, cal_buffers)

    # --- Kabelanteil berechnen ---
    if chip and ext:
        # Kreisarithmetik: Differenz der beiden Phasen
        cable_rad = float(np.angle(
            np.exp(1j * ext["phase_rad"]) / np.exp(1j * chip["phase_rad"])
        ))
        cable_deg = float(np.degrees(cable_rad))
    else:
        cable_rad = None
        cable_deg = None

    # --- Zusammenfassung ausgeben ---
    print("\n[CAL ] ── Zusammenfassung ────────────────────────────────────────────")
    if chip:
        print(f"[CAL ]   Chip-Offset   (Stufe 1) : {chip['phase_deg']:+.4f}°")
    if ext:
        print(f"[CAL ]   Gesamt-Offset (Stufe 2) : {ext['phase_deg']:+.4f}°")
    if cable_deg is not None:
        print(f"[CAL ]   Kabelanteil   (S2 - S1) : {cable_deg:+.4f}°")
        # Physische Interpretation
        c_eff   = 2e8   # typische Phasengeschwindigkeit Koaxkabel (0.66c)
        f_hz    = freq_hz + CAL_TONE_OFFSET_HZ
        delta_l = abs(cable_deg) / 360.0 * c_eff / f_hz * 100  # in cm
        print(f"[CAL ]   -> entspricht ~{delta_l:.1f} cm Laengenunterschied "
              f"(bei vp = 0.66c)")
    print(f"[CAL ] ─────────────────────────────────────────────────────────────")

    # Welche Phase wird zur Korrektur verwendet?
    if ext:
        corr_phase = ext["phase_rad"]
        corr_amp   = ext["amplitude_ratio"]
        corr_src   = "external"
    elif chip:
        corr_phase = chip["phase_rad"]
        corr_amp   = chip["amplitude_ratio"]
        corr_src   = "chip_only"
    else:
        corr_phase = 0.0
        corr_amp   = 1.0
        corr_src   = "none"

    return {
        # Stufe 1
        "chip_phase_rad":        chip.get("phase_rad"),
        "chip_phase_deg":        chip.get("phase_deg"),
        "chip_amplitude_ratio":  chip.get("amplitude_ratio"),
        "chip_snr_db":           chip.get("snr_db"),
        "chip_timestamp":        chip.get("timestamp"),
        # Stufe 2
        "total_phase_rad":       ext.get("phase_rad"),
        "total_phase_deg":       ext.get("phase_deg"),
        "total_amplitude_ratio": ext.get("amplitude_ratio"),
        "total_snr_db":          ext.get("snr_db"),
        "total_timestamp":       ext.get("timestamp"),
        # Kabelanteil
        "cable_phase_rad":       cable_rad,
        "cable_phase_deg":       cable_deg,
        # Korrekturstatus (wird spaeter gesetzt)
        "correction_applied":    False,
        "correction_source":     corr_src,
        "_corr_phase":           corr_phase,
        "_corr_amp":             corr_amp,
    }


def make_correction_factor(cal: dict) -> complex:
    """
    Berechnet komplexen Korrekturfaktor fuer RX1.
    exp(-j*phi) / amp  rotiert und skaliert RX1 auf RX0-Referenz.
    """
    phi = cal.get("_corr_phase", 0.0)
    amp = cal.get("_corr_amp",   1.0)
    if amp < 1e-6:
        return complex(1.0)
    return complex(np.exp(-1j * phi) / amp)


def _stop_tx(sdr: adi.ad9361) -> None:
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    try:
        sdr.tx_hardwaregain_chan0 = -89
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pluto-Konfiguration
# ---------------------------------------------------------------------------

def configure_pluto(uri: str, freq_hz: int, bw_hz: int, sr_sps: int,
                    gain_db: int, buffer_size: int) -> adi.ad9361:
    print(f"[INIT] Verbinde mit Pluto unter {uri} ...")
    sdr = adi.ad9361(uri=uri)

    sdr.rx_enabled_channels     = [0, 1]
    sdr.rx_lo                   = freq_hz
    sdr.sample_rate             = sr_sps
    sdr.rx_rf_bandwidth         = bw_hz
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0   = gain_db
    sdr.rx_hardwaregain_chan1   = gain_db
    sdr.rx_buffer_size          = buffer_size

    print(f"[INIT] Pluto konfiguriert:")
    print(f"       LO-Frequenz  : {freq_hz / 1e6:.6f} MHz")
    print(f"       Bandbreite   : {bw_hz   / 1e6:.3f} MHz")
    print(f"       Abtastrate   : {sr_sps  / 1e6:.3f} MSPS")
    print(f"       Verstaerkung : {gain_db} dB (manuell)")
    print(f"       Puffergroesse: {buffer_size} Samples")
    print(f"       Kanaele      : RX0 + RX1")
    return sdr


# ---------------------------------------------------------------------------
# Haupt-Aufnahmeschleife
# ---------------------------------------------------------------------------

def record(args: argparse.Namespace) -> None:
    os.makedirs(args.outdir, exist_ok=True)

    freq_hz       = int(args.freq_mhz        * 1e6)
    bw_hz         = int(args.bw_mhz          * 1e6)
    sr_sps        = int(args.samplerate_msps  * 1e6)
    block_samples = samples_for_duration(sr_sps, args.block_minutes * 60)
    scale         = 2048.0

    sdr = configure_pluto(args.uri, freq_hz, bw_hz, sr_sps,
                          args.gain, args.buffer_size)

    # --- Kalibrierung ---
    cal = calibrate(
        sdr          = sdr,
        sr_sps       = sr_sps,
        freq_hz      = freq_hz,
        cal_buffers  = args.cal_buffers,
        tx_gain_db   = DEFAULT_TX_GAIN_DB,
        skip_ext_cal = args.skip_ext_cal,
    )

    apply_correction = not args.no_correction
    cal["correction_applied"] = apply_correction

    if apply_correction:
        correction = make_correction_factor(cal)
        print(f"\n[REC ] Phasenkorrektur auf RX1 aktiv  "
              f"(Quelle: {cal['correction_source']})  "
              f"Δφ = {np.degrees(cal.get('_corr_phase', 0)):+.4f}°")
    else:
        correction = complex(1.0)
        print(f"\n[REC ] Phasenkorrektur deaktiviert (--no-correction). "
              f"Offsets nur in Meta-Datei.")

    # Einschwingen nach Kalibrierung
    print(f"[REC ] Einschwingpause (0.5 s) ...")
    time.sleep(0.5)
    for _ in range(4):
        sdr.rx()

    # --- Graceful-Shutdown ---
    running = [True]
    def _sigint(sig, frame):
        print("\n[STOP] Aufnahme wird beendet ...")
        running[0] = False
    signal.signal(signal.SIGINT,  _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    print(f"[REC ] Starte Aufnahme -> Ausgabe: {os.path.abspath(args.outdir)}")
    print(f"[REC ] Blockgroesse: {block_samples:,} Samples "
          f"= {args.block_minutes} min @ {sr_sps / 1e6:.3f} MSPS")
    print(f"[REC ] Format     : SigMF cf32_le (.sigmf-data + .sigmf-meta)")
    print( "[REC ] Abbrechen mit Ctrl+C\n")

    block_index    = 0
    buf            = [np.empty(0, dtype=np.complex64),
                      np.empty(0, dtype=np.complex64)]
    block_start_ts = datetime.now(tz=timezone.utc)

    while running[0]:
        try:
            rx_data = sdr.rx()
        except Exception as exc:
            print(f"[ERR ] Fehler beim Lesen: {exc}")
            time.sleep(0.1)
            continue

        if not isinstance(rx_data, (list, tuple)):
            rx_data = [rx_data, rx_data]

        for ch in range(2):
            samples = np.array(rx_data[ch], dtype=np.complex64) / scale
            if ch == 1:
                samples *= correction
            buf[ch] = np.concatenate((buf[ch], samples))

        while len(buf[0]) >= block_samples:
            ts_now = block_start_ts
            for ch in range(2):
                data_path = save_sigmf_block(
                    outdir      = args.outdir,
                    freq_hz     = freq_hz,
                    sr_sps      = sr_sps,
                    channel     = ch,
                    block_index = block_index,
                    ts          = ts_now,
                    gain_db     = args.gain,
                    iq_complex  = buf[ch][:block_samples],
                    cal         = cal,
                )
                size_mb = data_path.stat().st_size / 1024 / 1024
                print(f"[SAVE] Block {block_index:04d} | RX{ch} -> "
                      f"{data_path.name}  ({size_mb:.1f} MB)")

            for ch in range(2):
                buf[ch] = buf[ch][block_samples:]

            block_index    += 1
            block_start_ts  = datetime.now(tz=timezone.utc)

    print(f"\n[DONE] {block_index} Bloecke gespeichert in: "
          f"{os.path.abspath(args.outdir)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADALM Pluto Dual-RX SigMF Interferometer-Recorder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uri",             default=DEFAULT_URI)
    p.add_argument("--freq-mhz",        type=float, default=DEFAULT_FREQ_MHZ)
    p.add_argument("--bw-mhz",          type=float, default=DEFAULT_BW_MHZ)
    p.add_argument("--samplerate-msps", type=float, default=DEFAULT_SR_MSPS)
    p.add_argument("--block-minutes",   type=float, default=DEFAULT_BLOCK_MIN)
    p.add_argument("--gain",            type=int,   default=DEFAULT_GAIN_DB)
    p.add_argument("--outdir",          default=DEFAULT_OUTDIR)
    p.add_argument("--buffer-size",     type=int,   default=DEFAULT_BUFFER_SIZE)
    p.add_argument("--cal-buffers",     type=int,   default=DEFAULT_CAL_BUFFERS,
                   help="Puffer pro Kalibrierungsstufe")
    p.add_argument("--skip-ext-cal",    action="store_true",
                   help="Stufe 2 (externes Signal) ueberspringen – nur Chip-Offset")
    p.add_argument("--no-correction",   action="store_true",
                   help="Offsets nur loggen, nicht auf RX1 anwenden")
    return p.parse_args()


if __name__ == "__main__":
    record(parse_args())
 
