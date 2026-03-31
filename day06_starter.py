"""
BUILDCORED ORCAS — Day 06: BreathClock
========================================
Capture mic input, detect breathing rhythm from
audio amplitude, visualize breaths as a live waveform,
and compute breaths-per-minute.

Hardware concept: Microphone as Analog Sensor
Your mic is an ADC — it converts air pressure (analog)
into digital samples. The Butterworth low-pass filter
removes high-frequency noise (talking, music) and keeps
only the slow breath envelope. This is exactly how a
medical respiration sensor works.

YOUR TASK:
1. Tune the Butterworth filter cutoff (TODO #1)
2. Tune the breath detection threshold (TODO #2)
3. Run it: python day06_starter.py
4. Push to GitHub before midnight

TIP: Breathe slowly and deliberately near your mic.
The filter removes everything except slow amplitude changes.

CONTROLS:
- Close the plot window or press Ctrl+C to quit
"""

import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import collections

# ============================================================
# AUDIO SETUP
# ============================================================

RATE = 44100          # Sample rate (samples per second)
CHUNK = 1024          # Samples per audio frame
FORMAT = pyaudio.paFloat32
CHANNELS = 1

try:
    pa = pyaudio.PyAudio()

    # Find a working input device
    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_index = i
            print(f"Using mic: {info['name']}")
            break

    if device_index is None:
        print("ERROR: No microphone found.")
        print("Check your system audio settings.")
        sys.exit(1)

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
    print("Mic stream opened successfully.")

except Exception as e:
    print(f"ERROR opening microphone: {e}")
    print("\nFixes:")
    print("  Mac:   brew install portaudio && pip install pyaudio")
    print("  Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    print("  Win:   pip install pipwin && pipwin install pyaudio")
    sys.exit(1)


# ============================================================
# TODO #1: Butterworth Low-Pass Filter
# ============================================================
# The filter removes high-frequency content (voice, music, noise)
# and keeps only the slow amplitude changes caused by breathing.
#
# CUTOFF_HZ: frequencies above this are removed.
#   Human breathing is ~0.15 to 0.5 Hz (roughly 8-30 breaths/min).
#   Set cutoff to capture breathing but reject everything else.
#
#   Too low (0.1 Hz): even slow breaths get filtered out
#   Too high (2.0 Hz): noise passes through, false breath detections
#   Start with 0.5 Hz
#
# FILTER_ORDER: how aggressively the filter cuts off.
#   Higher = sharper cutoff but may distort the signal.
#   2 is a safe starting point.
#
CUTOFF_HZ = 0.5       # <-- Adjust this
FILTER_ORDER = 2

# Calculate filter coefficients
# Nyquist frequency = half the effective sample rate
# Our effective rate = RATE / CHUNK (we get one amplitude value per chunk)
effective_rate = RATE / CHUNK  # ~43 Hz
nyquist = effective_rate / 2

# Clamp cutoff to valid range
normalized_cutoff = min(CUTOFF_HZ / nyquist, 0.95)
b_coeff, a_coeff = butter(FILTER_ORDER, normalized_cutoff, btype='low')


# ============================================================
# TODO #2: Breath Detection Threshold
# ============================================================
# After filtering, we get a smooth envelope that rises when
# you breathe in/out and falls during pauses.
#
# BREATH_THRESHOLD: the minimum envelope level to count as
# a breath. Depends on your mic sensitivity and distance.
#
# Too low: ambient noise triggers false breaths
# Too high: real breaths don't register
#
# Watch the "Envelope" value on the plot and set this to
# roughly 50% of the peak value you see when breathing.
#
BREATH_THRESHOLD = 0.005   # <-- Adjust this (watch the plot to calibrate)


# ============================================================
# DATA BUFFERS
# ============================================================

HISTORY_LENGTH = 500    # How many data points to show on plot (~12 seconds)

# Raw amplitude history (before filter)
raw_history = collections.deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

# Filtered envelope history (after Butterworth)
envelope_history = collections.deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

# Filter state (keeps continuity between chunks)
filter_state = np.zeros(max(len(a_coeff), len(b_coeff)) - 1)

# Breath tracking
breath_times = []         # timestamps of detected breaths
is_above_threshold = False  # for edge detection (rising/falling)
current_bpm = 0.0


def compute_bpm():
    """Calculate breaths per minute from recent breath timestamps."""
    now = time.time()
    # Only count breaths from last 30 seconds
    recent = [t for t in breath_times if now - t < 30]
    breath_times.clear()
    breath_times.extend(recent)

    if len(recent) < 2:
        return 0.0

    # Average interval between breaths
    intervals = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    avg_interval = sum(intervals) / len(intervals)

    if avg_interval > 0:
        return 60.0 / avg_interval
    return 0.0


# ============================================================
# MATPLOTLIB SETUP
# ============================================================

fig, (ax_raw, ax_env) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("BreathClock — Day 06", fontsize=14, fontweight='bold')

# Raw waveform axis
ax_raw.set_xlim(0, HISTORY_LENGTH)
ax_raw.set_ylim(0, 0.05)
ax_raw.set_ylabel("Raw Amplitude")
ax_raw.set_title("Mic Input (RMS per chunk)")
line_raw, = ax_raw.plot([], [], color='#4fc3f7', linewidth=1)

# Filtered envelope axis
ax_env.set_xlim(0, HISTORY_LENGTH)
ax_env.set_ylim(0, 0.03)
ax_env.set_ylabel("Envelope")
ax_env.set_xlabel("Time →")
ax_env.set_title("Filtered Breath Envelope")
line_env, = ax_env.plot([], [], color='#66bb6a', linewidth=2)

# Threshold line
threshold_line = ax_env.axhline(y=BREATH_THRESHOLD, color='#ef5350',
                                 linestyle='--', linewidth=1, label='Threshold')
ax_env.legend(loc='upper right')

# BPM text
bpm_text = ax_env.text(0.02, 0.85, "BPM: --", transform=ax_env.transAxes,
                        fontsize=16, fontweight='bold', color='#66bb6a',
                        verticalalignment='top')

# Status text
status_text = ax_raw.text(0.02, 0.85, "Waiting...", transform=ax_raw.transAxes,
                           fontsize=12, color='white',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

plt.tight_layout()


# ============================================================
# ANIMATION UPDATE
# ============================================================

def update(frame_num):
    global filter_state, is_above_threshold, current_bpm

    try:
        # Read audio chunk
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.float32)

        # Compute RMS (root mean square) amplitude
        rms = np.sqrt(np.mean(samples ** 2))
        raw_history.append(rms)

        # Apply Butterworth filter to get smooth envelope
        # We filter the entire raw history each time for simplicity
        raw_array = np.array(raw_history)
        filtered, filter_state = lfilter(b_coeff, a_coeff, [rms], zi=filter_state)
        envelope_val = abs(filtered[0])
        envelope_history.append(envelope_val)

        # ---- BREATH DETECTION ----
        # Rising edge: envelope crosses above threshold
        if envelope_val > BREATH_THRESHOLD and not is_above_threshold:
            is_above_threshold = True
            breath_times.append(time.time())
            current_bpm = compute_bpm()

        # Falling edge: envelope drops below threshold
        elif envelope_val < BREATH_THRESHOLD * 0.7:  # Hysteresis!
            is_above_threshold = False

        # ---- UPDATE PLOT ----
        x_data = list(range(HISTORY_LENGTH))
        line_raw.set_data(x_data, list(raw_history))
        line_env.set_data(x_data, list(envelope_history))

        # Auto-scale y axis based on recent data
        raw_max = max(list(raw_history)[-100:]) if any(raw_history) else 0.01
        env_max = max(list(envelope_history)[-100:]) if any(envelope_history) else 0.01
        ax_raw.set_ylim(0, max(raw_max * 1.5, 0.005))
        ax_env.set_ylim(0, max(env_max * 1.5, BREATH_THRESHOLD * 2))

        # Update threshold line position (in case axis rescaled)
        threshold_line.set_ydata([BREATH_THRESHOLD])

        # Update text
        if current_bpm > 0:
            bpm_text.set_text(f"BPM: {current_bpm:.1f}")
        else:
            bpm_text.set_text("BPM: -- (breathe near mic)")

        breathing = "BREATH DETECTED" if is_above_threshold else "Listening..."
        breath_color = '#66bb6a' if is_above_threshold else 'white'
        status_text.set_text(breathing)
        status_text.set_color(breath_color)

    except Exception as e:
        status_text.set_text(f"Error: {e}")

    return line_raw, line_env, bpm_text, status_text, threshold_line


# ============================================================
# RUN
# ============================================================
print("\nBreathClock is running!")
print(f"Filter cutoff: {CUTOFF_HZ} Hz")
print(f"Breath threshold: {BREATH_THRESHOLD}")
print(f"Effective sample rate: {effective_rate:.1f} Hz")
print("\nBreathe slowly and deliberately near your mic.")
print("Watch the green envelope — each peak is a breath.")
print("Close the plot window to quit.\n")

try:
    ani = animation.FuncAnimation(fig, update, interval=int(1000 * CHUNK / RATE),
                                   blit=False, cache_frame_data=False)
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print(f"\nBreathClock ended. Final BPM: {current_bpm:.1f}")
    print("See you tomorrow for Day 07!")
