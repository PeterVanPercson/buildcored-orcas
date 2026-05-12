"""
BUILDCORED ORCAS — Day 30: OrcaOS
The capstone. Everything in one TUI shell.

Integrated components:
  - Gesture input (Week 1: hand detection via MediaPipe)
  - LLM reasoning (Week 2: qwen2.5:3b via ollama)
  - Sensor monitor (Week 3: mic RMS + CPU %)
  - Command shell (Week 4: full system integration)

Hardware concept: RTOS Architecture
Tasks = Textual workers. Queues = shared state.
Scheduler = Textual event loop. ISRs = sensor threads.
This is firmware — running on your laptop.

YOUR TASK:
1. Add one more component from your 30 days (TODO #1)
2. Tag a GitHub release v1.0 (TODO #2)

Run: python day30_starter.py
PREREQS: pip install textual
         ollama running with qwen2.5:3b
"""

import sys
import time
import threading
import queue
import subprocess
import platform
import collections

try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Header, Footer, Static, Input, Log,
        ProgressBar, Label, Button
    )
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.reactive import reactive
    from textual import work
    from textual.worker import Worker
    from textual.binding import Binding
except ImportError:
    print("pip install textual"); sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pyaudio
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    import cv2
    import mediapipe as mp
    HAS_GESTURE = True
except ImportError:
    HAS_GESTURE = False

LLM_MODEL = "qwen2.5:3b"


# ============================================================
# SHARED STATE (thread-safe)
# ============================================================

class OrcaState:
    """Central state store for all components."""
    def __init__(self):
        self.lock = threading.Lock()

        # Gesture
        self.gesture_label = "No hand"
        self.gesture_confidence = 0.0

        # Sensor
        self.mic_rms = 0.0
        self.cpu_pct = 0.0
        self.mem_pct = 0.0

        # LLM
        self.llm_response = ""
        self.llm_thinking = False

        # Log history
        self.event_log = collections.deque(maxlen=50)

    def log(self, msg):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            self.event_log.append(f"[{ts}] {msg}")


state = OrcaState()


# ============================================================
# BACKGROUND SENSOR THREAD
# ============================================================

def sensor_thread():
    """Read CPU, memory, and mic RMS continuously."""
    pa = None
    stream = None

    if HAS_AUDIO:
        try:
            pa = pyaudio.PyAudio()
            dev = None
            for i in range(pa.get_device_count()):
                if pa.get_device_info_by_index(i)['maxInputChannels'] > 0:
                    dev = i; break
            if dev is not None:
                stream = pa.open(
                    format=pyaudio.paFloat32, channels=1,
                    rate=16000, input=True,
                    input_device_index=dev,
                    frames_per_buffer=1600
                )
        except:
            pass

    while True:
        try:
            if HAS_PSUTIL:
                with state.lock:
                    state.cpu_pct = psutil.cpu_percent(interval=None)
                    state.mem_pct = psutil.virtual_memory().percent

            if stream:
                try:
                    data = stream.read(1600, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.float32)
                    with state.lock:
                        state.mic_rms = float(np.sqrt(np.mean(samples**2)))
                except:
                    pass

        except:
            pass

        time.sleep(0.1)


# ============================================================
# BACKGROUND GESTURE THREAD
# ============================================================

def gesture_thread():
    """Detect hand gestures from webcam."""
    if not HAS_GESTURE:
        state.gesture_label = "MediaPipe not installed"
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        state.gesture_label = "No webcam"
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    def classify_gesture(hand_lm):
        """Simple gesture classifier based on finger states."""
        lm = hand_lm.landmark

        # Finger tips and PIP joints (for open/closed detection)
        tips = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
        pips = [3, 6, 10, 14, 18]   # PIP joints

        fingers_up = []
        for tip, pip in zip(tips[1:], pips[1:]):
            fingers_up.append(lm[tip].y < lm[pip].y)

        # Thumb
        fingers_up.insert(0, lm[4].x < lm[3].x)

        count = sum(fingers_up)

        if count == 0: return "Fist ✊", 0.9
        if count == 5: return "Open Hand 🖐", 0.9
        if fingers_up[1] and not any(fingers_up[2:]): return "Point ☝️", 0.85
        if fingers_up[1] and fingers_up[2] and not any(fingers_up[3:]): return "Peace ✌️", 0.85
        if fingers_up[0] and not any(fingers_up[1:]): return "Thumbs Up 👍", 0.85
        return f"{count} fingers", 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            label, conf = classify_gesture(results.multi_hand_landmarks[0])
            with state.lock:
                if state.gesture_label != label:
                    state.log(f"Gesture: {label}")
                state.gesture_label = label
                state.gesture_confidence = conf
        else:
            with state.lock:
                state.gesture_label = "No hand"
                state.gesture_confidence = 0.0

        time.sleep(0.05)

    cap.release()


# ============================================================
# LLM QUERY
# ============================================================

def ask_llm(prompt):
    """Query local LLM. Returns response string."""
    system = (
        "You are OrcaOS, a hardware engineering assistant. "
        "You help with embedded systems, signal processing, and firmware. "
        "Reply concisely — 1-3 sentences max."
    )
    full_prompt = f"{system}\n\nUser: {prompt}\nOrcaOS:"
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL, full_prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[LLM timed out]"
    except Exception as e:
        return f"[Error: {e}]"


# ============================================================
# TEXTUAL TUI
# ============================================================

class GesturePanel(Static):
    """Widget showing current gesture detection."""

    DEFAULT_CSS = """
    GesturePanel {
        border: solid cyan;
        height: 8;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("🖐 Gesture Input", id="gesture-title")
        yield Label("Initializing...", id="gesture-label")
        yield Label("", id="gesture-conf")

    def on_mount(self):
        self.set_interval(0.2, self.refresh_gesture)

    def refresh_gesture(self):
        with state.lock:
            label = state.gesture_label
            conf = state.gesture_confidence
        self.query_one("#gesture-label", Label).update(f"  {label}")
        conf_bar = "█" * int(conf * 10)
        self.query_one("#gesture-conf", Label).update(
            f"  Confidence: {conf_bar} {conf*100:.0f}%"
        )


class SensorPanel(Static):
    """Widget showing live sensor channels."""

    DEFAULT_CSS = """
    SensorPanel {
        border: solid green;
        height: 8;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("📊 Sensor Monitor", id="sensor-title")
        yield Label("CPU:  --", id="cpu-label")
        yield Label("MEM:  --", id="mem-label")
        yield Label("MIC:  --", id="mic-label")

    def on_mount(self):
        self.set_interval(0.5, self.refresh_sensors)

    def refresh_sensors(self):
        with state.lock:
            cpu = state.cpu_pct
            mem = state.mem_pct
            mic = state.mic_rms

        cpu_bar = "█" * int(cpu / 10)
        mem_bar = "█" * int(mem / 10)
        mic_bar = "█" * min(10, int(mic * 200))

        self.query_one("#cpu-label", Label).update(
            f"  CPU: {cpu_bar:<10} {cpu:.0f}%"
        )
        self.query_one("#mem-label", Label).update(
            f"  MEM: {mem_bar:<10} {mem:.0f}%"
        )
        self.query_one("#mic-label", Label).update(
            f"  MIC: {mic_bar:<10} {mic:.4f}"
        )


class LLMPanel(Static):
    """Widget for LLM chat interface."""

    DEFAULT_CSS = """
    LLMPanel {
        border: solid yellow;
        height: 14;
        padding: 1;
    }
    #llm-response {
        height: 6;
        overflow-y: scroll;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("🤖 OrcaOS LLM", id="llm-title")
        yield Label("Ask a hardware engineering question:", id="llm-hint")
        yield Input(placeholder="e.g. What is PWM?", id="llm-input")
        yield Label("", id="llm-response")
        yield Label("", id="llm-status")

    def on_input_submitted(self, event: Input.Submitted):
        prompt = event.value.strip()
        if not prompt:
            return

        event.input.clear()
        self.query_one("#llm-status", Label).update("⏳ Thinking...")
        self.query_one("#llm-response", Label).update("")
        state.log(f"Q: {prompt[:40]}")

        # Run LLM in background worker
        self.run_llm(prompt)

    @work(thread=True)
    def run_llm(self, prompt: str):
        response = ask_llm(prompt)
        state.log(f"A: {response[:40]}")

        def update():
            self.query_one("#llm-response", Label).update(f"  {response}")
            self.query_one("#llm-status", Label).update("✓ Ready")

        self.app.call_from_thread(update)


class EventLog(Static):
    """Scrolling event log panel."""

    DEFAULT_CSS = """
    EventLog {
        border: solid $panel;
        height: 10;
        padding: 1;
        overflow-y: scroll;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("📋 Event Log", id="log-title")
        yield Log(id="log-widget", auto_scroll=True)

    def on_mount(self):
        self.set_interval(0.5, self.refresh_log)
        state.log("OrcaOS v1.0 started")
        state.log(f"Platform: {platform.system()}")
        state.log(f"LLM: {LLM_MODEL}")

    def refresh_log(self):
        log_widget = self.query_one("#log-widget", Log)
        with state.lock:
            events = list(state.event_log)

        # Write new events only
        if not hasattr(self, '_last_count'):
            self._last_count = 0

        new_events = events[self._last_count:]
        for event in new_events:
            log_widget.write_line(event)
        self._last_count = len(events)


# ============================================================
# TODO #1: Add one more component
# ============================================================
# Ideas — each maps to a previous day:
#
# class PWMPanel(Static):       ← Day 17 PWM visualization
# class MorsePanel(Static):     ← Day 28 tap-to-Morse
# class DepthPanel(Static):     ← Day 18 depth value readout
# class SoundPanel(Static):     ← Day 15 FFT band levels
# class CachePanel(Static):     ← Day 26 cache hit rate
#
# Template:
#
# class MyPanel(Static):
#     DEFAULT_CSS = """
#     MyPanel { border: solid magenta; height: 8; padding: 1; }
#     """
#     def compose(self) -> ComposeResult:
#         yield Label("🔧 My Component", id="my-title")
#         yield Label("", id="my-value")
#
#     def on_mount(self):
#         self.set_interval(0.5, self.refresh)
#
#     def refresh(self):
#         # Read from state or compute value
#         self.query_one("#my-value", Label).update("value here")
#
# Then add it to OrcaOS.compose() below.


# ============================================================
# MAIN APP
# ============================================================

class OrcaOS(App):
    """OrcaOS — the BUILDCORED ORCAS capstone shell."""

    CSS = """
    Screen {
        background: #0a0a0f;
    }
    Header {
        background: #0f7173;
        color: white;
    }
    Footer {
        background: #1a1a26;
    }
    Label {
        color: $text;
    }
    #main-container {
        padding: 1;
    }
    #top-row {
        height: auto;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]

    TITLE = "OrcaOS v1.0"
    SUB_TITLE = "BUILDCORED ORCAS — Day 30 Capstone"

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            # Top row: gesture + sensor side by side
            with Horizontal(id="top-row"):
                yield GesturePanel()
                yield SensorPanel()

            # Middle: LLM chat
            yield LLMPanel()

            # Bottom: event log
            yield EventLog()

            # TODO #1: Add your component here
            # yield MyPanel()

        yield Footer()

    def action_quit(self):
        self.exit()

    def action_clear_log(self):
        state.event_log.clear()
        state.log("Log cleared")

    def on_mount(self):
        state.log("All systems nominal")
        state.log(f"Gesture: {'active' if HAS_GESTURE else 'disabled'}")
        state.log(f"Audio: {'active' if HAS_AUDIO else 'disabled'}")


# ============================================================
# TODO #2: Tag a GitHub release
# ============================================================
# After your TUI is working and your screen recording is done:
#
# git add -A
# git commit -m "Day 30: OrcaOS capstone — 30 days complete"
# git tag v1.0
# git push origin main --tags
#
# This creates a v1.0 release on your GitHub repo.
# Include your best day's project as a release asset.
# Write a release description: what you built, what you learned.
#
# That's your portfolio proof. 30 commits. 30 projects. 1 release.


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 55)
    print("  🐋 OrcaOS — Day 30 Capstone")
    print("=" * 55)
    print()
    print(f"  Platform: {platform.system()}")
    print(f"  LLM:      {LLM_MODEL}")
    print(f"  Gesture:  {'✓' if HAS_GESTURE else '✗ (pip install opencv-python mediapipe)'}")
    print(f"  Audio:    {'✓' if HAS_AUDIO else '✗ (pip install pyaudio numpy)'}")
    print(f"  psutil:   {'✓' if HAS_PSUTIL else '✗ (pip install psutil)'}")
    print()

    # Start background threads
    threads = [
        threading.Thread(target=sensor_thread, daemon=True),
        threading.Thread(target=gesture_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    time.sleep(0.5)  # Let threads initialize

    # Launch Textual TUI
    app = OrcaOS()
    app.run()

    print("\n" + "=" * 55)
    print("  🐋 30 days. 30 projects. Done.")
    print()
    print("  Tag your release:")
    print("  git add -A")
    print('  git commit -m "Day 30: OrcaOS — 30 days complete"')
    print("  git tag v1.0")
    print("  git push origin main --tags")
    print()
    print("  See you in v2.0. 🔌")
    print("=" * 55)


if __name__ == "__main__":
    main()
