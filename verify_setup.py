#!/usr/bin/env python3
"""
BUILDCORED ORCAS — Environment Verification Script
Run this before Day 1 to confirm your setup is ready.

Usage:
    python verify_setup.py

This script checks for all required dependencies and hardware.
It uses ONLY the Python standard library — no external packages needed to run it.
"""

import sys
import os
import platform
import subprocess
import shutil
import importlib
import json


# --- Configuration ---

REQUIRED_PYTHON = (3, 10)

REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pygame", "pygame"),
    ("sounddevice", "sounddevice"),
    ("librosa", "librosa"),
    ("psutil", "psutil"),
    ("rich", "rich"),
    ("gitpython", "gitpython"),  # import name is "git" but pip name is gitpython
    ("PIL", "Pillow"),
    ("pyttsx3", "pyttsx3"),
    ("pynput", "pynput"),
    ("textual", "textual"),
    ("sklearn", "scikit-learn"),
]

# These are checked separately because import name != pip name
SPECIAL_IMPORT_MAP = {
    "gitpython": "git",
}

OLLAMA_MODELS = ["qwen2.5:3b", "moondream"]

# --- Color helpers (ANSI, no dependencies) ---

def supports_color():
    """Check if terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if platform.system() == "Windows":
        # Windows 10+ supports ANSI if we enable it
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("TERM") is not None
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = supports_color()

def green(text):
    return f"\033[92m{text}\033[0m" if USE_COLOR else text

def red(text):
    return f"\033[91m{text}\033[0m" if USE_COLOR else text

def yellow(text):
    return f"\033[93m{text}\033[0m" if USE_COLOR else text

def bold(text):
    return f"\033[1m{text}\033[0m" if USE_COLOR else text

def dim(text):
    return f"\033[2m{text}\033[0m" if USE_COLOR else text


# --- OS Detection ---

def get_os():
    """Return 'mac', 'windows', or 'linux'."""
    s = platform.system().lower()
    if s == "darwin":
        return "mac"
    elif s == "windows":
        return "windows"
    else:
        return "linux"


# --- Check functions ---

def check_python_version():
    """Check Python >= 3.10."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= REQUIRED_PYTHON:
        return True, f"Python {version_str}"
    else:
        fixes = {
            "mac": "brew install python@3.12  OR  download from https://python.org",
            "windows": "Download from https://python.org (check 'Add to PATH' during install)",
            "linux": "sudo apt-get install python3.12  OR  use pyenv",
        }
        return False, f"Python {version_str} (need 3.10+)\n    Fix: {fixes[get_os()]}"


def check_package(import_name, pip_name):
    """Check if a Python package is importable."""
    actual_import = SPECIAL_IMPORT_MAP.get(pip_name, import_name)
    try:
        importlib.import_module(actual_import)
        return True, pip_name
    except ImportError:
        return False, f"{pip_name} not found\n    Fix: pip install {pip_name}"


def check_webcam():
    """Check if a webcam is accessible via OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return True, "Webcam accessible (index 0)"
            else:
                return False, "Webcam opened but cannot read frames\n    Fix: Close other apps using the webcam, then retry"
        else:
            # Try index 1
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return True, "Webcam accessible (index 1)"
            cap.release()
            fixes = {
                "mac": "Check System Preferences > Security & Privacy > Camera. Grant terminal/IDE access.",
                "windows": "Check Settings > Privacy > Camera. Ensure camera access is enabled.",
                "linux": "Check if /dev/video0 exists: ls /dev/video*. Install: sudo apt-get install v4l-utils",
            }
            return False, f"No webcam detected\n    Fix: {fixes[get_os()]}"
    except Exception as e:
        return False, f"Webcam check failed: {e}"


def check_microphone():
    """Check if a microphone is accessible via pyaudio."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        found_input = False
        for i in range(device_count):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                found_input = True
                break
        pa.terminate()
        if found_input:
            return True, "Microphone accessible"
        else:
            return False, "No input device found\n    Fix: Check system audio settings and ensure a mic is connected"
    except ImportError:
        fixes = {
            "mac": "brew install portaudio && pip install pyaudio",
            "windows": "pip install pyaudio  (if that fails: pip install pipwin && pipwin install pyaudio)",
            "linux": "sudo apt-get install portaudio19-dev python3-pyaudio && pip install pyaudio",
        }
        return False, f"pyaudio not installed\n    Fix: {fixes[get_os()]}"
    except Exception as e:
        fixes = {
            "mac": "brew install portaudio && pip install pyaudio --force-reinstall",
            "windows": "pip install pyaudio  (if that fails: pip install pipwin && pipwin install pyaudio)",
            "linux": "sudo apt-get install portaudio19-dev && pip install pyaudio --force-reinstall",
        }
        return False, f"Microphone check failed: {e}\n    Fix: {fixes[get_os()]}"


def check_ollama_running():
    """Check if ollama is installed and the server is reachable."""
    # Check if ollama binary exists
    if not shutil.which("ollama"):
        fixes = {
            "mac": "Download from https://ollama.com or: brew install ollama",
            "windows": "Download from https://ollama.com",
            "linux": "curl -fsSL https://ollama.com/install.sh | sh",
        }
        return False, f"ollama not found in PATH\n    Fix: {fixes[get_os()]}"

    # Check if ollama server is running by listing models
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, "ollama is running"
        else:
            return False, (
                "ollama is installed but server is not running\n"
                "    Fix: Open a separate terminal and run: ollama serve\n"
                "    Then re-run this script."
            )
    except subprocess.TimeoutExpired:
        return False, "ollama server not responding (timed out)\n    Fix: Restart ollama: ollama serve"
    except Exception as e:
        return False, f"ollama check failed: {e}\n    Fix: Start ollama: ollama serve"


def check_ollama_model(model_name):
    """Check if a specific ollama model is pulled."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False, f"Cannot check models — ollama server not running"

        # Parse the model list
        output = result.stdout.lower()
        # ollama list output has model names in the first column
        # Model names may appear as "qwen2.5:3b" or "qwen2.5:3b-instruct" etc.
        model_base = model_name.split(":")[0].lower()
        model_tag = model_name.split(":")[1].lower() if ":" in model_name else ""

        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            # First column is the model name
            listed_model = line.strip().split()[0] if line.strip().split() else ""
            if model_tag:
                if listed_model.startswith(f"{model_base}:{model_tag}"):
                    return True, f"Model {model_name} available"
            else:
                if listed_model.startswith(model_base):
                    return True, f"Model {model_name} available"

        return False, f"Model {model_name} not found\n    Fix: ollama pull {model_name}"

    except Exception as e:
        return False, f"Model check failed: {e}"


def check_git():
    """Check if git is installed and configured."""
    if not shutil.which("git"):
        fixes = {
            "mac": "Install Xcode CLI tools: xcode-select --install  OR  brew install git",
            "windows": "Download from https://git-scm.com (use default settings)",
            "linux": "sudo apt-get install git",
        }
        return False, f"git not found\n    Fix: {fixes[get_os()]}"

    # Check if git user.name is configured
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5
        )
        name = result.stdout.strip()
        if name:
            return True, f"Git configured (user: {name})"
        else:
            return False, (
                "Git installed but user.name not configured\n"
                '    Fix: git config --global user.name "Your Name"\n'
                '          git config --global user.email "you@example.com"'
            )
    except Exception:
        return True, "Git installed (could not check config)"


# --- Main ---

def main():
    print()
    print(bold("=" * 60))
    print(bold("  BUILDCORED ORCAS — Environment Verification"))
    print(bold("=" * 60))
    print()
    print(f"  Platform:  {platform.system()} {platform.release()}")
    print(f"  Machine:   {platform.machine()}")
    print(f"  Python:    {sys.executable}")
    print()
    print(bold("-" * 60))
    print()

    results = []
    total = 0
    passed = 0

    def run_check(name, check_fn, *args):
        nonlocal total, passed
        total += 1
        success, detail = check_fn(*args)
        results.append((name, success, detail))
        if success:
            passed += 1
            print(f"  {green('[PASS]')} {name}")
            print(f"         {dim(detail)}")
        else:
            print(f"  {red('[FAIL]')} {name}")
            for line in detail.split("\n"):
                print(f"         {line}")
        print()

    # Core checks
    run_check("Python 3.10+", check_python_version)
    run_check("Git", check_git)

    # Package checks
    print(bold("  --- Python Packages ---"))
    print()
    pkg_pass = 0
    pkg_fail = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        total += 1
        success, detail = check_package(import_name, pip_name)
        results.append((f"Package: {pip_name}", success, detail))
        if success:
            passed += 1
            pkg_pass += 1
        else:
            pkg_fail.append(pip_name)

    if not pkg_fail:
        print(f"  {green('[PASS]')} All {pkg_pass} packages installed")
        print()
    else:
        print(f"  {green('[PASS]')} {pkg_pass}/{pkg_pass + len(pkg_fail)} packages installed")
        print()
        for pkg in pkg_fail:
            print(f"  {red('[FAIL]')} {pkg} not found")
            print(f"         Fix: pip install {pkg}")
            print()
        if len(pkg_fail) > 1:
            install_cmd = "pip install " + " ".join(pkg_fail)
            print(f"  {yellow('Install all missing:')} {install_cmd}")
            print()

    # Hardware checks
    print(bold("  --- Hardware ---"))
    print()
    run_check("Webcam", check_webcam)
    run_check("Microphone", check_microphone)

    # ollama checks
    print(bold("  --- ollama (Local AI) ---"))
    print()
    run_check("ollama running", check_ollama_running)

    # Only check models if ollama is running
    ollama_ok = any(name == "ollama running" and success for name, success, _ in results)
    if ollama_ok:
        for model in OLLAMA_MODELS:
            run_check(f"ollama model: {model}", check_ollama_model, model)
    else:
        for model in OLLAMA_MODELS:
            total += 1
            results.append((f"ollama model: {model}", False, "Skipped — ollama not running"))
            print(f"  {red('[SKIP]')} ollama model: {model}")
            print(f"         Skipped — fix ollama first, then run: ollama pull {model}")
            print()

    # Summary
    print(bold("=" * 60))
    print()
    if passed == total:
        print(f"  {green(bold(f'ALL CHECKS PASSED ({passed}/{total})'))} ")
        print()
        print(f"  {green('Your environment is ready for BUILDCORED ORCAS.')}")
        print(f"  Screenshot this output and post it in your squad channel.")
    else:
        failed = total - passed
        print(f"  {red(bold(f'{failed} CHECK(S) FAILED'))}  ({passed}/{total} passed)")
        print()
        print(f"  Fix the issues above and run this script again.")
        print(f"  If you're stuck, post in #tech-support with this format:")
        print()
        print(f"    OS: {platform.system()} | Check: [which one failed] | Error: [exact message] | Tried: [what you did]")

    print()
    print(bold("=" * 60))
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
