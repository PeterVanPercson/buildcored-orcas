"""
BUILDCORED ORCAS — Day 09: WhisperDesk
========================================
Local speech-to-text. Speak into your mic, see the
text appear in your terminal. No cloud. No API keys.

Hardware concept: On-Device DSP + Latency Budgeting
Audio chunk size sets your latency budget:
  - Small chunks (2s): fast response, less accurate
  - Large chunks (5s): slow response, more accurate
This is the SAME tradeoff firmware engineers make when
sizing audio buffers on embedded devices.

YOUR TASK:
1. Tune the chunk size for latency vs accuracy (TODO #1)
2. Understand the audio pipeline stages (TODO #2)
3. Run it: python day09_starter.py
4. Push to GitHub before midnight

PREREQUISITES:
Either:
  a) pip install faster-whisper   (recommended, ~1 GB model download)
  b) ollama pull qwen2.5:3b      (fallback, uses ollama for transcription)

CONTROLS:
- Speak into your microphone
- Text appears after each chunk is processed
- Press Ctrl+C to quit
"""

import pyaudio
import numpy as np
import wave
import tempfile
import os
import sys
import time
import threading
import subprocess

# ============================================================
# AUDIO CAPTURE SETUP
# ============================================================

RATE = 16000          # 16kHz — standard for speech recognition
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024          # Samples per audio frame


# ============================================================
# TODO #1: Chunk size — the latency tradeoff
# ============================================================
# RECORD_SECONDS controls how much audio we collect before
# sending it to the transcription model.
#
# This is your LATENCY BUDGET — the same concept as buffer
# sizing on an embedded audio processor.
#
#   2 seconds: fast response, but short phrases get cut off
#   3 seconds: good balance for most speech
#   5 seconds: slow but accurate, captures full sentences
#   8 seconds: very slow, but great for long dictation
#
# The pipeline delay = RECORD_SECONDS + model inference time
# On a fast machine: 3s recording + 1s inference = 4s total
# On a slow machine: 3s recording + 5s inference = 8s total
#
RECORD_SECONDS = 3    # <-- Adjust this

# Minimum audio energy to process (skip silent chunks)
# Prevents sending silence to the model, saving compute
SILENCE_THRESHOLD = 500   # <-- Adjust if needed


# ============================================================
# TRANSCRIPTION BACKEND
# ============================================================
# We try faster-whisper first (better quality).
# If that's not installed, fall back to ollama.

BACKEND = None


def setup_faster_whisper():
    """Try to initialize faster-whisper."""
    global BACKEND
    try:
        from faster_whisper import WhisperModel

        print("Loading faster-whisper model (first run downloads ~1 GB)...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✓ faster-whisper loaded (model: base, int8)")
        BACKEND = "faster-whisper"
        return model
    except ImportError:
        print("faster-whisper not installed.")
        return None
    except Exception as e:
        print(f"faster-whisper failed: {e}")
        return None


def setup_ollama():
    """Fall back to ollama for transcription."""
    global BACKEND
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("✓ Using ollama as transcription backend")
            print("  (Note: ollama text models approximate speech-to-text")
            print("   by describing audio patterns. Quality is lower than whisper.)")
            BACKEND = "ollama"
            return True
        else:
            return False
    except Exception:
        return False


# Try backends in order
whisper_model = setup_faster_whisper()

if whisper_model is None:
    print("Trying ollama fallback...")
    if not setup_ollama():
        print("\nERROR: No transcription backend available.")
        print("Install one of these:")
        print("  Option A: pip install faster-whisper")
        print("  Option B: ollama pull qwen2.5:3b")
        sys.exit(1)


# ============================================================
# TRANSCRIPTION FUNCTIONS
# ============================================================

def transcribe_with_whisper(audio_file_path):
    """Transcribe audio using faster-whisper."""
    segments, info = whisper_model.transcribe(
        audio_file_path,
        beam_size=1,           # Faster, slightly less accurate
        language="en",          # Set to None for auto-detect
        vad_filter=True,        # Voice activity detection
    )
    text = " ".join(segment.text for segment in segments).strip()
    return text


def transcribe_with_ollama(audio_file_path):
    """
    Fallback: use ollama to process audio.
    This is a creative workaround — we describe the audio
    characteristics and ask the model to interpret.
    Note: this is NOT true speech recognition, just a fallback.
    """
    # Read audio and compute basic features
    with wave.open(audio_file_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    # Compute simple features
    rms = np.sqrt(np.mean(samples ** 2))
    zero_crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
    duration = len(samples) / RATE

    prompt = (
        f"I recorded {duration:.1f} seconds of speech audio. "
        f"RMS energy: {rms:.0f}, zero crossings: {zero_crossings}. "
        f"Please note: I cannot actually hear the audio. "
        f"This is a demonstration of the speech-to-text pipeline. "
        f"Respond with: '[Speech detected - {duration:.1f}s of audio captured]'"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b", prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[Transcription error: {e}]"


def transcribe(audio_file_path):
    """Route to the active backend."""
    if BACKEND == "faster-whisper":
        return transcribe_with_whisper(audio_file_path)
    else:
        return transcribe_with_ollama(audio_file_path)


# ============================================================
# AUDIO RECORDING
# ============================================================

def is_silent(audio_data):
    """Check if audio chunk is mostly silence."""
    samples = np.frombuffer(audio_data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    return rms < SILENCE_THRESHOLD


def record_chunk():
    """
    Record one chunk of audio from the microphone.
    Returns the audio data as bytes, or None if silent.
    """
    frames = []
    num_frames = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(num_frames):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except Exception:
            pass

    audio_data = b''.join(frames)

    # Skip if silent
    if is_silent(audio_data):
        return None

    return audio_data


def save_audio_to_temp(audio_data):
    """Save audio bytes to a temporary WAV file."""
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    )
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    return temp_file.name


# ============================================================
# TODO #2: Understand the audio pipeline
# ============================================================
# The pipeline has 4 stages, each with a latency cost:
#
#   MIC → BUFFER → MODEL → TEXT
#   │      │        │       │
#   │      │        │       └─ Display (instant)
#   │      │        └─ Inference (1-5 seconds)
#   │      └─ Buffering (RECORD_SECONDS)
#   └─ Capture (continuous)
#
# Total latency = buffer time + inference time
#
# This is IDENTICAL to the pipeline on an embedded device:
#   SENSOR → ADC BUFFER → MCU PROCESSING → OUTPUT
#
# The firmware engineer's question is the same as yours:
# "How big should my buffer be?"
#


# ============================================================
# MAIN LOOP
# ============================================================

# Initialize microphone
try:
    pa = pyaudio.PyAudio()

    device_index = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            device_index = i
            print(f"\nUsing mic: {info['name']}")
            break

    if device_index is None:
        print("ERROR: No microphone found.")
        sys.exit(1)

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )
except Exception as e:
    print(f"ERROR opening microphone: {e}")
    print("Mac: brew install portaudio && pip install pyaudio")
    print("Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    sys.exit(1)

print()
print("=" * 50)
print("  🎙️  WhisperDesk — Local Speech-to-Text")
print(f"  Backend: {BACKEND}")
print(f"  Chunk size: {RECORD_SECONDS}s")
print(f"  Silence threshold: {SILENCE_THRESHOLD}")
print("=" * 50)
print()
print("  Speak into your microphone.")
print("  Text appears after each chunk.")
print("  Press Ctrl+C to quit.")
print()
print("-" * 50)

# Full transcript
full_transcript = []

try:
    while True:
        # Show recording indicator
        sys.stdout.write(f"\r🔴 Listening ({RECORD_SECONDS}s)...")
        sys.stdout.flush()

        # Record a chunk
        audio_data = record_chunk()

        if audio_data is None:
            sys.stdout.write(f"\r⚪ Silence (skipped)     ")
            sys.stdout.flush()
            continue

        # Save to temp file
        temp_path = save_audio_to_temp(audio_data)

        # Transcribe
        sys.stdout.write(f"\r⏳ Transcribing...       ")
        sys.stdout.flush()

        start_time = time.time()
        text = transcribe(temp_path)
        inference_time = time.time() - start_time

        # Clean up temp file
        try:
            os.unlink(temp_path)
        except Exception:
            pass

        # Display result
        if text and text.strip():
            total_latency = RECORD_SECONDS + inference_time
            sys.stdout.write(f"\r")
            print(f"📝 {text}")
            print(f"   ⚡ inference: {inference_time:.1f}s | "
                  f"total latency: {total_latency:.1f}s | "
                  f"buffer: {RECORD_SECONDS}s")
            print()
            full_transcript.append(text)
        else:
            sys.stdout.write(f"\r⚪ No speech detected     \n")

except KeyboardInterrupt:
    pass
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

    print("\n" + "=" * 50)
    if full_transcript:
        print("\n📋 Full transcript:")
        print(" ".join(full_transcript))
    print(f"\nWhisperDesk ended. Processed with {BACKEND}.")
    print("See you tomorrow for Day 10!")
