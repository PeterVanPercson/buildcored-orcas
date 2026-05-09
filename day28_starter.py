"""
BUILDCORED ORCAS — Day 28: MorseDecoder
Tap spacebar in Morse code. Decode to text.
LLM responds. Response encoded back to Morse.

Hardware concept: Serial Bit Encoding + Pulse-Width Timing
Morse = manual UART. Dot = short pulse. Dash = long pulse.
Letter gap = stop bit. Word gap = idle line.
Same state machine a UART receiver uses to decode bits.

YOUR TASK:
1. Implement auto-baud detection (TODO #1)
2. Add the Morse encoder for LLM responses (TODO #2)
3. Run: python day28_starter.py

CONTROLS:
- SPACE (hold) → input signal (dot or dash based on duration)
- Release SPACE → end of element
- Wait 3x dot → letter gap (ends current letter)
- Wait 7x dot → word gap (ends current word, sends to LLM)
- ESC → quit

TIPS:
- Start slow: 500ms dot, 1500ms dash
- The auto-detect (TODO #1) will calibrate timing for you
- Click the terminal window first so pynput captures keys
"""

import time
import sys
import threading
import collections

try:
    from pynput import keyboard
    BACKEND = "pynput"
except ImportError:
    print("pip install pynput")
    print("Mac: also grant Accessibility permission to Terminal")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)

import subprocess

MODEL = "qwen2.5:3b"

# ============================================================
# MORSE CODE TABLE
# ============================================================

MORSE_TO_CHAR = {
    ".-": "A",   "-...": "B", "-.-.": "C", "-..": "D",
    ".": "E",    "..-.": "F", "--.": "G",  "....": "H",
    "..": "I",   ".---": "J", "-.-": "K",  ".-..": "L",
    "--": "M",   "-.": "N",   "---": "O",  ".--.": "P",
    "--.-": "Q", ".-.": "R",  "...": "S",  "-": "T",
    "..-": "U",  "...-": "V", ".--": "W",  "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?",
    "-..-.": "/",  "---...": ":",
}

CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}


def encode_morse(text):
    """Encode text to Morse code string."""
    result = []
    for char in text.upper():
        if char == " ":
            result.append("/")
        elif char in CHAR_TO_MORSE:
            result.append(CHAR_TO_MORSE[char])
    return " ".join(result)


def decode_morse_sequence(sequence):
    """Decode a dot/dash sequence string like '.-' to a character."""
    return MORSE_TO_CHAR.get(sequence, "?")


# ============================================================
# TIMING CONFIGURATION
# ============================================================

# TODO #1: Auto-baud detection
# ============================================================
# In real UART, baud rate detection measures the first few
# transitions to infer the bit period.
# For Morse: measure the first 5 tap durations and set
# DOT_DURATION to the median of short taps.
#
# The class AutoBaud below is a skeleton — complete it.
# After calibration, DOT_DURATION updates automatically.
#

class AutoBaud:
    """Detect dot timing from the first few taps."""
    def __init__(self, calibration_taps=5):
        self.calibration_taps = calibration_taps
        self.tap_durations = []
        self.calibrated = False
        self.dot_duration = None

    def add_tap(self, duration_ms):
        """Add a tap duration during calibration."""
        if self.calibrated:
            return

        self.tap_durations.append(duration_ms)

        if len(self.tap_durations) >= self.calibration_taps:
            self._calibrate()

    def _calibrate(self):
        """Set dot duration from shortest taps (assumed to be dots)."""
        sorted_durations = sorted(self.tap_durations)
        # Shortest half are probably dots
        dot_candidates = sorted_durations[:max(1, len(sorted_durations) // 2)]
        self.dot_duration = sum(dot_candidates) / len(dot_candidates)
        self.calibrated = True
        console.print(f"\n[green]✓ Auto-calibrated: dot = {self.dot_duration:.0f}ms[/green]")

    @property
    def is_calibrated(self):
        return self.calibrated


auto_baud = AutoBaud(calibration_taps=6)

# Default timing (used until auto-calibrated)
DOT_DURATION = 300        # ms — tap shorter than this = dot
DASH_MULTIPLIER = 2.5     # tap longer than DOT * this = dash
LETTER_GAP_MULTIPLIER = 3  # silence longer than DOT * this = end of letter
WORD_GAP_MULTIPLIER = 7    # silence longer than DOT * this = end of word


def get_dot_duration():
    """Get current dot duration (auto-calibrated or default)."""
    if auto_baud.calibrated and auto_baud.dot_duration:
        return auto_baud.dot_duration
    return DOT_DURATION


# ============================================================
# TIMING STATE MACHINE
# ============================================================

class MorseState:
    IDLE = "IDLE"
    KEY_DOWN = "KEY_DOWN"
    BETWEEN_ELEMENTS = "BETWEEN_ELEMENTS"
    LETTER_GAP = "LETTER_GAP"


class MorseDecoder:
    """
    State machine for Morse decoding.

    States:
      IDLE → key pressed → KEY_DOWN
      KEY_DOWN → key released → BETWEEN_ELEMENTS
      BETWEEN_ELEMENTS → short silence → next element
      BETWEEN_ELEMENTS → letter gap → commit letter
      Letter committed → word gap → commit word
    """

    def __init__(self):
        self.state = MorseState.IDLE
        self.key_down_time = None
        self.key_up_time = None

        self.current_sequence = ""     # dots and dashes for current letter
        self.current_word = ""         # letters for current word
        self.decoded_text = ""         # full session text

        self.elements = []             # list of (type, duration_ms)
        self.lock = threading.Lock()

        # Gap checker thread
        self._running = True
        self._gap_thread = threading.Thread(
            target=self._gap_checker, daemon=True
        )
        self._gap_thread.start()

    def key_pressed(self):
        with self.lock:
            now = time.time()
            self.key_down_time = now
            self.key_up_time = None
            self.state = MorseState.KEY_DOWN

    def key_released(self):
        with self.lock:
            now = time.time()
            if self.key_down_time is None:
                return

            duration_ms = (now - self.key_down_time) * 1000
            self.key_up_time = now
            self.key_down_time = None

            # Auto-baud calibration
            auto_baud.add_tap(duration_ms)

            # Classify as dot or dash
            dot = get_dot_duration()
            if duration_ms < dot * DASH_MULTIPLIER:
                self.current_sequence += "."
                self.elements.append(("dot", duration_ms))
            else:
                self.current_sequence += "-"
                self.elements.append(("dash", duration_ms))

            self.state = MorseState.BETWEEN_ELEMENTS

    def _gap_checker(self):
        """Background thread: detect letter and word gaps from silence."""
        while self._running:
            time.sleep(0.05)

            with self.lock:
                if self.state != MorseState.BETWEEN_ELEMENTS:
                    continue
                if self.key_up_time is None:
                    continue

                silence_ms = (time.time() - self.key_up_time) * 1000
                dot = get_dot_duration()

                if silence_ms >= dot * WORD_GAP_MULTIPLIER:
                    # Word gap — commit letter then word
                    if self.current_sequence:
                        char = decode_morse_sequence(self.current_sequence)
                        self.current_word += char
                        self.current_sequence = ""

                    if self.current_word:
                        word = self.current_word
                        self.decoded_text += word + " "
                        self.current_word = ""
                        self.state = MorseState.IDLE
                        # Signal word completion (handled in main loop)
                        self._on_word_complete(word)

                elif silence_ms >= dot * LETTER_GAP_MULTIPLIER:
                    # Letter gap — commit current letter
                    if self.current_sequence:
                        char = decode_morse_sequence(self.current_sequence)
                        self.current_word += char
                        self.current_sequence = ""
                        self.state = MorseState.IDLE

    def _on_word_complete(self, word):
        """Called when a word is committed. Override or extend."""
        pass  # Main loop polls decoded_text

    def stop(self):
        self._running = False


# ============================================================
# LLM INTEGRATION
# ============================================================

def ask_llm(decoded_text):
    """Send decoded Morse text to LLM. Returns response string."""
    prompt = (
        f"A person sent you this message via Morse code: '{decoded_text.strip()}'\n"
        f"Reply in exactly 1 sentence. Be friendly and brief."
    )
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except:
        return "CAN NOT CONNECT TO LOCAL MODEL"


# ============================================================
# TODO #2: Morse encoder for LLM responses
# ============================================================
# After the LLM responds, encode its response back to Morse
# and display it so the student can practice reading.
#
# `encode_morse(text)` is already implemented above.
# Display the result with timing hints:
#   . = short flash (100ms)
#   - = long flash (300ms)
#   gap = silence (100ms between elements, 300ms between letters)
#
# Use rich to display with color:
#   [cyan].[/cyan] and [cyan]---[/cyan]
#

def display_morse_response(llm_response):
    """Encode LLM response to Morse and display it visually."""
    morse = encode_morse(llm_response)

    console.print(f"\n[bold]LLM says:[/bold] {llm_response}")
    console.print(f"[bold]In Morse:[/bold]")

    # Display each character's Morse code
    words = morse.split(" / ")
    for word_morse in words:
        letters = word_morse.split(" ")
        display_parts = []
        for letter_code in letters:
            colored = ""
            for symbol in letter_code:
                if symbol == ".":
                    colored += "[cyan]·[/cyan]"
                elif symbol == "-":
                    colored += "[cyan]—[/cyan]"
            display_parts.append(colored)
        console.print("  " + "  ".join(display_parts))

    console.print(f"\n[dim]{morse}[/dim]\n")


# ============================================================
# MAIN
# ============================================================

def main():
    console.print("\n[bold cyan]📡 MorseDecoder — Day 28[/bold cyan]")
    console.print("[dim]Tap spacebar in Morse code. Decoded text feeds the LLM.[/dim]\n")
    console.print(f"  [bold]Timing (default):[/bold]")
    console.print(f"  Dot threshold: {DOT_DURATION}ms")
    console.print(f"  Dash: >{DOT_DURATION * DASH_MULTIPLIER:.0f}ms")
    console.print(f"  Letter gap: >{DOT_DURATION * LETTER_GAP_MULTIPLIER:.0f}ms")
    console.print(f"  Word gap: >{DOT_DURATION * WORD_GAP_MULTIPLIER:.0f}ms\n")
    console.print("  [bold]Morse reference:[/bold]")
    console.print("  A=.-  B=-...  E=.  I=..  S=...  T=-")
    console.print("  M=--  O=---  H=....  R=.-.  N=-.")
    console.print()
    console.print("  [dim]Click this terminal window first to capture keys.[/dim]")
    console.print("  [dim]Hold SPACE for dots/dashes. ESC to quit.[/dim]\n")

    decoder = MorseDecoder()
    last_decoded_len = 0
    quit_flag = threading.Event()

    def on_press(key):
        if key == keyboard.Key.space:
            decoder.key_pressed()
        elif key == keyboard.Key.esc:
            quit_flag.set()
            return False

    def on_release(key):
        if key == keyboard.Key.space:
            decoder.key_released()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    console.print("[green]Ready. Start tapping.[/green]\n")

    try:
        while not quit_flag.is_set():
            time.sleep(0.1)

            with decoder.lock:
                seq = decoder.current_sequence
                word_so_far = decoder.current_word
                full_text = decoder.decoded_text

            # Show live state
            display_parts = []
            if full_text:
                display_parts.append(f"[white]{full_text}[/white]")
            if word_so_far:
                display_parts.append(f"[yellow]{word_so_far}[/yellow]")
            if seq:
                display_parts.append(f"[cyan]{seq}[/cyan]")

            if display_parts:
                status = " ".join(display_parts)
                console.print(f"\r  {status}          ", end="")

            # Check if a new word was completed
            if len(full_text) > last_decoded_len and full_text.strip():
                last_decoded_len = len(full_text)
                new_word = full_text.strip().split()[-1] if full_text.strip() else ""

                if new_word:
                    console.print(f"\n\n[green]✓ Word committed: '{new_word}'[/green]")

                    # Check if it's time to ask LLM
                    # (ask after every complete word or accumulate — student's choice)
                    console.print(f"[dim]Full text so far: '{full_text.strip()}'[/dim]")
                    console.print("[dim]Waiting for more (word gap = send to LLM)[/dim]")

            # Send to LLM after longer pause (detected by gap checker)
            # We poll for new complete sentences (text ending in multiple words)
            words_count = len(full_text.strip().split()) if full_text.strip() else 0
            if words_count >= 2 and len(full_text) > last_decoded_len + 5:
                last_decoded_len = len(full_text) + 100  # Prevent re-triggering
                console.print(f"\n[bold]Sending to LLM:[/bold] '{full_text.strip()}'")
                response = ask_llm(full_text)
                display_morse_response(response)
                decoder.decoded_text = ""
                last_decoded_len = 0

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        decoder.stop()

    console.print(f"\n\n[bold]Final decoded text:[/bold] '{decoder.decoded_text.strip()}'")
    console.print("\nMorseDecoder ended. See you tomorrow for Day 29!")


if __name__ == "__main__":
    main()
