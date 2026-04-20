"""
BUILDCORED ORCAS — Day 16: EchoKiller
Adaptive FIR filter for acoustic echo cancellation.

Hardware concept: Acoustic Echo Cancellation (AEC)
This exact algorithm runs inside every speakerphone DSP chip,
hearing aid, and smart speaker. You're building in software
what TI, Cirrus Logic, and Qualcomm ship as silicon.

The LMS (Least Mean Squares) algorithm:
- Predicts the echo by running reference signal through an FIR filter
- Subtracts predicted echo from the mixed signal
- Adjusts filter coefficients to reduce the error
- Repeats — the filter "learns" the room's echo profile

YOUR TASK:
1. Tune the FIR filter order (TODO #1)
2. Tune the LMS learning rate (TODO #2)
3. Understand what the coefficients represent (TODO #3)

Run: python day16_starter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import sounddevice as sd
    HAS_PLAYBACK = True
except ImportError:
    HAS_PLAYBACK = False


SAMPLE_RATE = 16000  # Standard for speech


# ============================================================
# SYNTHETIC ECHO GENERATION
# If the student doesn't provide an audio file, we generate
# a synthetic speech-like signal and add artificial echo.
# This guarantees everyone can run the day.
# ============================================================

def generate_synthetic_speech(duration=3.0, sample_rate=SAMPLE_RATE):
    """Generate speech-like audio using frequency-modulated sine waves."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Speech-like formants (roughly vowel frequencies)
    signal = np.zeros_like(t)

    # Create word-like bursts with pauses
    for burst_start in [0.2, 0.9, 1.6, 2.3]:
        burst_len = 0.5
        mask = (t >= burst_start) & (t < burst_start + burst_len)

        # Each "word" has a different fundamental + harmonics
        fundamental = np.random.uniform(100, 200)
        word = np.zeros_like(t)
        for harmonic in range(1, 5):
            word += np.sin(2 * np.pi * fundamental * harmonic * t) / harmonic

        # Envelope (fade in/out of the burst)
        envelope = np.zeros_like(t)
        envelope[mask] = np.sin(np.pi * (t[mask] - burst_start) / burst_len)

        signal += word * envelope

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.6
    return signal.astype(np.float32)


def add_synthetic_echo(signal, delay_ms=150, decay=0.5, sample_rate=SAMPLE_RATE):
    """Add a single echo to the signal."""
    delay_samples = int(sample_rate * delay_ms / 1000)
    echo = np.zeros_like(signal)
    if delay_samples < len(signal):
        echo[delay_samples:] = signal[:-delay_samples] * decay
    return signal + echo


# ============================================================
# LOAD AUDIO
# ============================================================

def load_or_generate():
    """Try to load a .wav file from current dir. Otherwise generate one."""
    # Look for any .wav file in current directory
    wav_files = [f for f in os.listdir(".") if f.lower().endswith(".wav")]

    if wav_files and HAS_SOUNDFILE:
        path = wav_files[0]
        print(f"📂 Loading: {path}")
        try:
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data[:, 0]  # Mono
            # Resample if needed (simple decimation)
            if sr != SAMPLE_RATE:
                ratio = sr / SAMPLE_RATE
                indices = (np.arange(int(len(data) / ratio)) * ratio).astype(int)
                data = data[indices]
            print(f"   Loaded {len(data)/SAMPLE_RATE:.1f}s of audio")
            return data.astype(np.float32), True
        except Exception as e:
            print(f"   Load failed: {e}")

    print("📡 No .wav file found — generating synthetic speech")
    clean = generate_synthetic_speech()
    return clean, False


# ============================================================
# TODO #1: FIR filter order
# ============================================================
# The filter order = number of coefficients = number of past
# samples the filter looks at to predict the echo.
#
# The filter needs to span AT LEAST the echo delay.
# Echo delay of 150ms at 16kHz = 2400 samples.
# So order must be >= 2400 to capture that echo.
#
# Higher order = can model longer/more complex echoes
# Lower order = faster, but can't handle long reverb
#
FILTER_ORDER = 2500   # <-- Adjust this


# ============================================================
# TODO #2: LMS learning rate (mu)
# ============================================================
# The learning rate controls how fast the filter adapts.
#
# Too high (0.1):   unstable, filter oscillates, may diverge
# Too low  (0.001): extremely slow, barely removes echo
# Good range:       0.01 - 0.05
#
# This is the same tradeoff as gradient descent in ML — and it
# IS gradient descent, just with one step per sample.
#
LEARNING_RATE = 0.02   # <-- Adjust this


# ============================================================
# LMS ADAPTIVE FILTER
# ============================================================

def lms_filter(reference, mixed, filter_order, mu):
    """
    Run LMS adaptive filter.

    Args:
        reference: the clean signal (what went INTO the room)
        mixed:     the signal containing echo (what came OUT of the mic)
        filter_order: number of FIR taps
        mu:        learning rate

    Returns:
        error_signal: mixed - predicted_echo  (this is the clean output)
        coefficients: the final learned FIR taps
    """
    N = len(mixed)

    # Initialize filter coefficients to zero
    # They will adapt to represent the room's impulse response
    w = np.zeros(filter_order, dtype=np.float32)

    # Output (cleaned) signal
    error = np.zeros(N, dtype=np.float32)

    # Reference buffer (past samples of the clean signal)
    ref_buffer = np.zeros(filter_order, dtype=np.float32)

    for n in range(N):
        # Shift reference buffer and add newest sample
        ref_buffer[1:] = ref_buffer[:-1]
        ref_buffer[0] = reference[n]

        # Predict echo: FIR filter output
        predicted_echo = np.dot(w, ref_buffer)

        # Error = what the mic heard - what we predicted
        # If the filter is perfect, this is the clean signal
        e = mixed[n] - predicted_echo
        error[n] = e

        # Update coefficients (gradient descent on squared error)
        # Normalize by reference energy for stability (NLMS variant)
        norm = np.dot(ref_buffer, ref_buffer) + 1e-6
        w = w + (mu / norm) * e * ref_buffer

    return error, w


# ============================================================
# TODO #3: Understand the coefficients
# ============================================================
# After LMS finishes, the `coefficients` array IS the room's
# impulse response. Each coefficient represents the amplitude
# of the echo at that time delay.
#
# If you see a big spike at index 2400, that means the room
# produces an echo 2400 samples (~150 ms) after the direct sound.
# Multiple spikes = multiple reflections from different walls.
#
# When you plot the coefficients, you're literally looking at
# the acoustic signature of the room.
#


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(clean, echoed, cleaned, coefficients, sample_rate):
    """Side-by-side comparison plots."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 9))
    fig.suptitle("EchoKiller — Day 16", fontsize=14, fontweight='bold')

    t = np.arange(len(clean)) / sample_rate

    # Plot 1: Clean reference
    axes[0].plot(t, clean, color='#43a047', linewidth=0.8)
    axes[0].set_title("1. Clean Reference (input to room)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Echoed (what the mic heard)
    axes[1].plot(t, echoed, color='#e53935', linewidth=0.8)
    axes[1].set_title("2. With Echo (mic input)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Cleaned (after LMS)
    axes[2].plot(t, cleaned, color='#1e88e5', linewidth=0.8)
    axes[2].set_title("3. After EchoKiller (LMS output)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Learned filter coefficients (the room's impulse response!)
    tap_times = np.arange(len(coefficients)) / sample_rate * 1000  # ms
    axes[3].stem(tap_times[::5], coefficients[::5], basefmt=" ",
                 linefmt='#ff6f00', markerfmt='o')
    axes[3].set_title(f"4. Learned FIR Coefficients — the room's impulse response ({len(coefficients)} taps)")
    axes[3].set_xlabel("Delay (ms)")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 50)
    print("  🔇 EchoKiller — Adaptive FIR Echo Cancellation")
    print("=" * 50)
    print()

    # Load or generate audio
    clean, from_file = load_or_generate()

    # Add echo (if from generator) OR assume the file already has echo
    if not from_file:
        print("📢 Adding synthetic echo (150ms delay, 50% decay)")
        echoed = add_synthetic_echo(clean, delay_ms=150, decay=0.5)
    else:
        # For a real recording, we can't separate clean vs mixed without extra info
        # For this demo, still apply synthetic echo so LMS has something to learn
        print("📢 Adding extra synthetic echo for demonstration")
        echoed = add_synthetic_echo(clean, delay_ms=150, decay=0.5)

    print(f"\n⚙️  LMS Parameters:")
    print(f"   Filter order: {FILTER_ORDER} taps")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Audio length: {len(clean)/SAMPLE_RATE:.1f}s ({len(clean)} samples)")

    print("\n🧠 Running LMS adaptive filter...")
    import time
    start = time.time()
    cleaned, coefficients = lms_filter(clean, echoed, FILTER_ORDER, LEARNING_RATE)
    elapsed = time.time() - start
    print(f"   Done in {elapsed:.1f}s")

    # Report residual error
    echo_energy = np.mean(echoed ** 2)
    cleaned_energy = np.mean(cleaned ** 2)
    clean_energy = np.mean(clean ** 2)

    print(f"\n📊 Energy analysis:")
    print(f"   Clean signal:   {clean_energy:.6f}")
    print(f"   With echo:      {echo_energy:.6f}")
    print(f"   After filter:   {cleaned_energy:.6f}")
    reduction_db = 10 * np.log10(echo_energy / max(cleaned_energy, 1e-9))
    print(f"   Noise reduction: {reduction_db:.1f} dB")

    # Optional playback
    if HAS_PLAYBACK:
        print("\n🔊 Playback order: clean → echoed → cleaned")
        print("   (press Ctrl+C to skip)")
        try:
            print("   Playing CLEAN...")
            sd.play(clean, SAMPLE_RATE); sd.wait()
            print("   Playing ECHOED...")
            sd.play(echoed, SAMPLE_RATE); sd.wait()
            print("   Playing CLEANED...")
            sd.play(cleaned, SAMPLE_RATE); sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("   Playback skipped.")

    # Save cleaned output
    if HAS_SOUNDFILE:
        sf.write("cleaned_output.wav", cleaned, SAMPLE_RATE)
        sf.write("echoed_input.wav", echoed, SAMPLE_RATE)
        print("\n💾 Saved: cleaned_output.wav, echoed_input.wav")

    # Show plots
    print("\n📈 Displaying waveforms and filter coefficients...")
    plot_results(clean, echoed, cleaned, coefficients, SAMPLE_RATE)

    print("\nSee you tomorrow for Day 17!")


if __name__ == "__main__":
    main()
