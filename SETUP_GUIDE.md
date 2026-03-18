# BUILDCORED ORCAS — Environment Setup Guide

Complete this setup **before Day 1**. A failed install on launch morning is a morale event, not just a technical one.

**Deadline:** You must run `verify_setup.py` and post a passing screenshot in your squad channel before the challenge starts.

---

## Step 1: Install Core Software

### Python 3.10+

| Platform | Install |
|----------|---------|
| **macOS** | `brew install python@3.12` or download from [python.org](https://python.org) |
| **Windows** | Download from [python.org](https://python.org). **Check "Add Python to PATH" during install.** |
| **Linux** | `sudo apt-get install python3.12 python3.12-venv python3-pip` or use [pyenv](https://github.com/pyenv/pyenv) |

Verify: `python --version` (or `python3 --version` on some systems)

### Git

| Platform | Install |
|----------|---------|
| **macOS** | `xcode-select --install` (installs git) or `brew install git` |
| **Windows** | Download from [git-scm.com](https://git-scm.com). Use default settings. |
| **Linux** | `sudo apt-get install git` |

Verify: `git --version`

Configure git (if you haven't):
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### ollama (Local AI Runtime)

ollama runs AI models on your machine — no cloud, no API keys, no cost.

| Platform | Install |
|----------|---------|
| **macOS** | Download from [ollama.com](https://ollama.com) or `brew install ollama` |
| **Windows** | Download from [ollama.com](https://ollama.com) |
| **Linux** | `curl -fsSL https://ollama.com/install.sh | sh` |

Verify: `ollama --version`

**Start the server** (keep this running in a separate terminal):
```bash
ollama serve
```

**Pull the required models** (this downloads ~2-4 GB total):
```bash
ollama pull qwen2.5:3b
ollama pull moondream
```

Verify models are pulled: `ollama list`

### OBS Studio (for screen recordings on Advanced/Expert days)

Download from [obsproject.com](https://obsproject.com). Free, all platforms.

Alternatively: macOS built-in (Cmd+Shift+5), Windows Game Bar (Win+G).

You won't need this until Day 10, but install it now so it's ready.

### Code Editor

[VS Code](https://code.visualstudio.com/) is recommended for its integrated terminal. Use whatever you're comfortable with.

---

## Step 2: Install Python Packages

Run this single command:

```bash
pip install opencv-python mediapipe numpy scipy matplotlib pygame sounddevice pyaudio librosa psutil rich gitpython Pillow pyttsx3 pynput faster-whisper chromadb sentence-transformers PyMuPDF textual scikit-learn
```

### Platform-Specific Issues

**macOS:**
```bash
# Required for pyaudio:
brew install portaudio

# Optional — MLX acceleration for Apple Silicon (not required):
pip install mlx mlx-lm
```

**Windows:**
```bash
# If pyaudio install fails:
pip install pipwin
pipwin install pyaudio

# You also need Microsoft C++ Build Tools for some packages:
# Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Linux:**
```bash
# Required for pyaudio:
sudo apt-get install portaudio19-dev python3-pyaudio

# Required for tkinter (used in some projects):
sudo apt-get install python3-tk

# Volume control (used in Day 3):
# Verify pactl or amixer is available:
pactl --version  # or: amixer --version

# Webcam check:
ls /dev/video*
# If no device, install: sudo apt-get install v4l-utils
```

---

## Step 3: Verify Your Webcam and Microphone

Your laptop's built-in webcam and microphone are your primary sensors for this challenge. Verify they work:

**Webcam:** Open any video call app or Photo Booth (Mac) / Camera (Windows) and confirm you see yourself.

**Microphone:** Open your system audio settings and confirm the input level meter moves when you speak.

If you're using an external USB webcam or microphone, make sure it's plugged in when you run the verification script.

---

## Step 4: Run the Verification Script

```bash
git clone https://github.com/YOUR_USERNAME/buildcored-orcas.git
cd buildcored-orcas
python verify_setup.py
```

The script checks everything: Python version, packages, webcam, microphone, ollama, git config. Each check shows PASS or FAIL with specific fix instructions.

**All checks must pass before Day 1.**

Screenshot your results and post them in your squad channel on Discord.

---

## Step 5: If Something Fails

1. Read the fix instruction printed by the script — it's OS-specific.
2. Run the fix command.
3. Run `verify_setup.py` again.
4. If it still fails, post in `#tech-support` on Discord with this format:

```
OS: [Mac/Windows/Linux] | Check: [which one failed] | Error: [exact message] | Tried: [what you already attempted]
```

Do not wait until the last day. Fix issues early.

---

## What Each Package Is For

Not every package is used every day. Here's why each one is in the install:

| Package | Used For |
|---------|----------|
| opencv-python | Webcam capture, image processing (Weeks 1-4) |
| mediapipe | Face, hand, and pose detection (Weeks 1, 4) |
| numpy | Array math, signal processing (every week) |
| scipy | Filters, FFT, signal processing (Weeks 1, 3) |
| matplotlib | Plotting, visualization (Weeks 1-4) |
| pygame | Audio playback, graphics (Weeks 1, 3) |
| sounddevice | Audio capture and playback (Weeks 1-3) |
| pyaudio | Microphone input (Weeks 1-3) |
| librosa | Audio analysis and manipulation (Week 1) |
| psutil | System monitoring, CPU/battery (Week 3) |
| rich | Terminal UI formatting (Weeks 2-4) |
| gitpython | Git integration (Week 2) |
| Pillow | Image handling (Weeks 2, 4) |
| pyttsx3 | Text-to-speech (Week 4) |
| pynput | Keyboard/mouse input (Weeks 1, 3, 4) |
| faster-whisper | Speech-to-text (Week 2) |
| chromadb | Vector database for RAG (Week 4) |
| sentence-transformers | Text embeddings (Week 4) |
| PyMuPDF | PDF parsing (Week 4) |
| textual | Terminal UI framework (Week 4) |
| scikit-learn | Machine learning (Week 4) |
