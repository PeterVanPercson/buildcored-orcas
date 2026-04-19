"""
BUILDCORED ORCAS — Day 15: AudioScope
Live FFT spectrum analyzer — your mic visualized.

Hardware concept: Spectrum analysis and filter banks.
The exact math a hardware spectrum analyzer IC runs.
Every audio EQ, every RF analyzer, every vibration
sensor uses this FFT pipeline.

YOUR TASK:
1. Tune the frequency band ranges (TODO #1)
2. Add peak frequency detection (TODO #2)

Run: python day15_starter.py
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# ============================================================
# AUDIO SETUP
# ============================================================

RATE = 44100          # Sample rate (Hz)
CHUNK = 2048          # Samples per FFT window — power of 2
FORMAT = pyaudio.paFloat32
CHANNELS = 1

try:
    pa = pyaudio.PyAudio()

    # Find an input device
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_index = i
            print(f"Using mic: {info['name']}")
            break

    if device_index is None:
        print("ERROR: No microphone found.")
        sys.exit(1)

    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
except Exception as e:
    print(f"ERROR opening mic: {e}")
    print("  Mac:   brew install portaudio && pip install pyaudio")
    print("  Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    print("  Win:   pip install pipwin && pipwin install pyaudio")
    sys.exit(1)


# ============================================================
# FFT THEORY (quick reminder)
# ============================================================
# FFT takes a time-domain signal (amplitude over time) and
# converts it to frequency-domain (amplitude per frequency).
#
# Input:  N audio samples at RATE Hz
# Output: N/2 frequency bins from 0 Hz to RATE/2 (Nyquist)
#
# Our bins: 1024 (because CHUNK/2 = 2048/2)
# Our max freq: 22050 Hz (RATE/2)
# Bin resolution: RATE/CHUNK = 21.5 Hz per bin
# ============================================================

# Pre-compute the frequency for each FFT bin
freqs = np.fft.rfftfreq(CHUNK, d=1.0/RATE)


# ============================================================
# TODO #1: Frequency band ranges
# ============================================================
# Audio engineers divide the spectrum into named bands.
# Each band has different perceptual/technical meaning.
#
# Default bands below are standard. Try changing them:
#   - Split "mid" into low-mid / high-mid
#   - Add "presence" (4-6 kHz) — where vocals cut through
#   - Add "air" (10+ kHz) — the shimmer on cymbals
#
# Format: (name, low_hz, high_hz, color_hex)
#
BANDS = [
    ("Sub-bass", 20,   60,    "#e53935"),   # Red — felt more than heard
    ("Bass",     60,   250,   "#fb8c00"),   # Orange — kick drum, bass guitar
    ("Low-mid",  250,  500,   "#fdd835"),   # Yellow — body of sound
    ("Mid",      500,  2000,  "#43a047"),   # Green — vocals, snare
    ("High-mid", 2000, 4000,  "#00acc1"),   # Cyan — presence, clarity
    ("Treble",   4000, 12000, "#1e88e5"),   # Blue — brightness, cymbals
    ("Air",      12000, 20000, "#8e24aa"),  # Purple — shimmer
]


def band_level(fft_magnitudes, freq_array, band):
    """Average FFT magnitude within a frequency band."""
    name, low, high, color = band
    mask = (freq_array >= low) & (freq_array < high)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(fft_magnitudes[mask]))


# ============================================================
# TODO #2: Peak detection
# ============================================================
# Find the single loudest frequency in the current spectrum.
# This is useful because most sounds have a "fundamental"
# frequency — a voice's pitch, a guitar string, a bird call.
#
# Returns the frequency in Hz of the loudest bin.
# Ignore very low frequencies (< 50 Hz) — they're mostly noise.
#
def find_peak_frequency(fft_magnitudes, freq_array, min_freq=50):
    """Find the frequency with the highest magnitude."""
    valid = freq_array >= min_freq
    if not valid.any():
        return 0.0
    valid_mags = fft_magnitudes.copy()
    valid_mags[~valid] = 0
    peak_idx = np.argmax(valid_mags)
    return float(freq_array[peak_idx])


# ============================================================
# MATPLOTLIB SETUP
# ============================================================

fig, (ax_spec, ax_bands) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("AudioScope — Day 15 | Speak or play music", fontsize=13)

# Spectrum plot (log-scale x-axis to see low freqs clearly)
ax_spec.set_xscale("log")
ax_spec.set_xlim(20, 20000)
ax_spec.set_ylim(0, 0.05)
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_ylabel("Magnitude")
ax_spec.set_title("Frequency Spectrum")
ax_spec.grid(True, which="both", alpha=0.3)

spectrum_line, = ax_spec.plot([], [], color='#4fc3f7', linewidth=1)

# Band level bar chart
band_names = [b[0] for b in BANDS]
band_colors = [b[3] for b in BANDS]
bars = ax_bands.bar(band_names, [0] * len(BANDS), color=band_colors)
ax_bands.set_ylim(0, 0.05)
ax_bands.set_ylabel("Average Magnitude")
ax_bands.set_title("Frequency Bands")
ax_bands.grid(True, axis='y', alpha=0.3)

# Peak frequency text
peak_text = ax_spec.text(
    0.02, 0.92, "Peak: --",
    transform=ax_spec.transAxes, fontsize=11, fontweight='bold',
    color='white',
    bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8)
)

plt.tight_layout()


# ============================================================
# ANIMATION UPDATE
# ============================================================

def update(frame_num):
    try:
        # Read audio chunk
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.float32)

        # Apply window (reduces spectral leakage)
        windowed = samples * np.hanning(len(samples))

        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_result) / CHUNK  # Normalize

        # Update spectrum line
        spectrum_line.set_data(freqs, magnitudes)

        # Auto-scale y axis based on recent peak
        peak_mag = max(magnitudes.max(), 0.001)
        ax_spec.set_ylim(0, peak_mag * 1.2)

        # Compute band levels
        levels = [band_level(magnitudes, freqs, band) for band in BANDS]
        for bar, level in zip(bars, levels):
            bar.set_height(level)

        # Auto-scale bands
        max_band = max(max(levels), 0.001)
        ax_bands.set_ylim(0, max_band * 1.2)

        # Peak detection
        peak_freq = find_peak_frequency(magnitudes, freqs)
        if peak_freq > 0:
            peak_text.set_text(f"Peak: {peak_freq:.0f} Hz")
        else:
            peak_text.set_text("Peak: --")

    except Exception as e:
        peak_text.set_text(f"Error: {e}")

    return spectrum_line, peak_text, *bars


# ============================================================
# RUN
# ============================================================
print("\nAudioScope is running!")
print(f"Sample rate: {RATE} Hz | FFT size: {CHUNK} | Resolution: {RATE/CHUNK:.1f} Hz/bin")
print(f"Nyquist limit: {RATE/2} Hz")
print(f"Bands: {', '.join(b[0] for b in BANDS)}")
print("\nTry: whistle, clap, speak, play music")
print("Close the plot window to quit.\n")

try:
    ani = animation.FuncAnimation(
        fig, update,
        interval=int(1000 * CHUNK / RATE),
        blit=False, cache_frame_data=False
    )
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("\nAudioScope ended. See you tomorrow for Day 16!")
