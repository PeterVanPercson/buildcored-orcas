"""
BUILDCORED ORCAS — Day 10: TerminalBrain
==========================================
Wrap a terminal command. Capture its stdout/stderr live.
When errors appear, ask a local LLM to suggest a fix
and display it inline.

Hardware concept: Interrupt-Driven Feedback Loop
Hardware watchdog timers monitor system state and trigger
recovery handlers on failure. Your wrapper is the watchdog.
The LLM is the recovery handler. The "interrupt" fires
when stderr produces an error pattern.

YOUR TASK:
1. Improve the error detection pattern (TODO #1)
2. Tune the LLM prompt for better fix suggestions (TODO #2)
3. Add pattern caching to avoid repeated LLM calls (TODO #3 - bonus)
4. Run it: python day10_starter.py <command>
5. Push to GitHub + screen recording before midnight

PREREQUISITES:
- ollama running: ollama serve
- Model pulled: ollama pull qwen2.5:3b

USAGE:
    python day10_starter.py python broken_script.py
    python day10_starter.py ls /nonexistent
    python day10_starter.py pip install nonexistent_package_xyz

CONTROLS:
- Wrapper runs the command
- stdout shows in white
- stderr shows in red
- LLM suggestions show in cyan
- Press Ctrl+C to quit
"""

import sys
import os
import platform
import subprocess
import threading
import queue
import time
import re
import argparse


# ============================================================
# CROSS-PLATFORM PROCESS WRAPPING
# This is the hard part. pty works on Unix but not Windows.
# We detect the OS and use the right approach.
# You don't need to change this section.
# ============================================================

IS_WINDOWS = platform.system() == "Windows"

if not IS_WINDOWS:
    try:
        import pty
        import select
        HAS_PTY = True
    except ImportError:
        HAS_PTY = False
else:
    HAS_PTY = False


# ============================================================
# COLORS (ANSI escape codes — work in most modern terminals)
# ============================================================

class Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


def color_text(text, color):
    return f"{color}{text}{Color.RESET}"


# ============================================================
# OLLAMA INTERFACE
# ============================================================

MODEL = "qwen2.5:3b"


def check_ollama():
    """Verify ollama is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False, "ollama not running. Run: ollama serve"
        if "qwen2.5" not in result.stdout.lower():
            return False, "Model missing. Run: ollama pull qwen2.5:3b"
        return True, "ok"
    except FileNotFoundError:
        return False, "ollama not installed. Get it from https://ollama.com"
    except Exception as e:
        return False, str(e)


def ask_llm_for_fix(error_text):
    """
    Send an error to the LLM and get a fix suggestion.
    This is the 'recovery handler' in our watchdog analogy.
    """
    prompt = build_llm_prompt(error_text)

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        suggestion = result.stdout.strip()
        return suggestion
    except subprocess.TimeoutExpired:
        return "[LLM timeout — error too complex]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ============================================================
# TODO #2: Tune the LLM prompt
# ============================================================
# The system prompt determines whether the LLM gives useful
# fixes or vague advice. A bad prompt = "try checking your code".
# A good prompt = a specific command or code change.
#
# Key things a good prompt should do:
# - Tell the model to be concise (not write paragraphs)
# - Ask for ONE specific fix, not a list
# - Mention common error categories (import, syntax, missing file)
# - Tell it to format the answer briefly
#

def build_llm_prompt(error_text):
    """Build the prompt sent to the LLM."""
    return f"""You are a terminal error fixer. A command just produced this error:

{error_text}

Give ONE specific fix in 1-2 sentences. If a command would fix it, show the exact command.
Be concise. No greetings, no explanations of what the error means — just the fix.
"""


# ============================================================
# TODO #1: Error detection
# ============================================================
# We need to decide WHEN a stderr line is actually an error
# (worth sending to the LLM) vs noise (warnings, debug info).
#
# Calling the LLM on every stderr line is too slow.
# We use pattern matching to filter for real errors.
#
# Add more patterns as you discover them in your testing.
#

ERROR_PATTERNS = [
    # Python errors
    r"Traceback \(most recent call last\)",
    r"Error:",
    r"Exception:",
    r"ModuleNotFoundError",
    r"ImportError",
    r"NameError",
    r"SyntaxError",
    r"TypeError",
    r"ValueError",
    r"KeyError",
    r"AttributeError",
    r"FileNotFoundError",

    # Shell errors
    r"command not found",
    r"No such file or directory",
    r"permission denied",
    r"cannot access",

    # Generic
    r"FAILED",
    r"FATAL",

    # TODO: Add more patterns you encounter!
]

ERROR_REGEX = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


def is_error_line(line):
    """Check if a line of stderr looks like a real error."""
    if not line or not line.strip():
        return False
    return bool(ERROR_REGEX.search(line))


# ============================================================
# TODO #3 (BONUS): Pattern caching
# ============================================================
# If the same error happens repeatedly, calling the LLM each
# time wastes compute. Cache fixes by error pattern.
#
# Simple approach: dict mapping error signature → cached fix.
# Error signature = first error keyword + module/file mentioned.
#
# A real watchdog implementation would do this — you don't
# need to call the recovery handler every time the same fault
# repeats.
#

fix_cache = {}


def get_cached_fix(error_text):
    """Look up a cached fix for similar errors."""
    # Simple cache key: first 80 chars of the error
    key = error_text[:80].strip()
    return fix_cache.get(key)


def cache_fix(error_text, fix):
    """Store a fix for future identical errors."""
    key = error_text[:80].strip()
    fix_cache[key] = fix


# ============================================================
# THREADED I/O — separate threads for stdout and stderr
# This prevents the deadlock that kills naive implementations.
# ============================================================

def reader_thread(stream, output_queue, stream_name):
    """Read from a stream and push lines to a queue."""
    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            output_queue.put((stream_name, line))
    except Exception as e:
        output_queue.put(("error", f"[reader thread error: {e}]\n"))
    finally:
        try:
            stream.close()
        except Exception:
            pass


# ============================================================
# MAIN COMMAND WRAPPER
# ============================================================

def run_with_brain(command):
    """
    Run a command, capture stdout/stderr live, analyze errors.
    """
    print(color_text(f"\n┌─ TerminalBrain wrapping: ", Color.CYAN), end="")
    print(color_text(" ".join(command), Color.BOLD))
    print(color_text("│ stdout = white | stderr = red | brain = cyan", Color.DIM))
    print(color_text("└─" + "─" * 50, Color.CYAN))
    print()

    # Start the subprocess with separate stderr/stdout pipes
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
    except FileNotFoundError:
        print(color_text(f"Command not found: {command[0]}", Color.RED))
        # Even this is an error worth analyzing!
        suggestion = ask_llm_for_fix(f"command not found: {command[0]}")
        print(color_text(f"\n🧠 Brain: {suggestion}\n", Color.CYAN))
        return
    except Exception as e:
        print(color_text(f"Failed to start process: {e}", Color.RED))
        return

    # Queue for combined output
    output_queue = queue.Queue()

    # Two reader threads — one for each stream
    stdout_thread = threading.Thread(
        target=reader_thread,
        args=(process.stdout, output_queue, "stdout"),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=reader_thread,
        args=(process.stderr, output_queue, "stderr"),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    # Buffer for accumulating multi-line errors (e.g. tracebacks)
    error_buffer = []
    in_error_block = False
    error_count = 0
    llm_calls = 0
    cache_hits = 0

    # Process output until command finishes
    while True:
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except queue.Empty:
            # Check if process is still alive
            if process.poll() is not None and output_queue.empty():
                break
            continue

        if stream_name == "stdout":
            # Print stdout in white
            print(color_text(line.rstrip(), Color.WHITE))

            # If we were collecting an error, the error block ended
            if in_error_block and error_buffer:
                handle_error_block(error_buffer)
                error_buffer = []
                in_error_block = False

        elif stream_name == "stderr":
            # Print stderr in red
            print(color_text(line.rstrip(), Color.RED))

            # Check if this looks like an error
            if is_error_line(line):
                in_error_block = True
                error_count += 1

            if in_error_block:
                error_buffer.append(line)

                # Send error to brain after we've collected a few lines
                # (or after timeout — handled by the next iteration)

    # Wait for any final output
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    # Process any remaining error buffer
    if error_buffer:
        handle_error_block(error_buffer)

    # Final summary
    print()
    print(color_text("─" * 52, Color.DIM))
    exit_code = process.returncode
    status_color = Color.GREEN if exit_code == 0 else Color.RED
    print(color_text(f"Exit code: {exit_code}", status_color))
    print(color_text(f"Errors detected: {error_count}", Color.DIM))
    print(color_text(f"LLM calls: {llm_calls} | cache hits: {cache_hits}", Color.DIM))


def handle_error_block(lines):
    """Send a collected error block to the LLM for analysis."""
    error_text = "".join(lines).strip()
    if not error_text:
        return

    # Check cache first
    cached = get_cached_fix(error_text)
    if cached:
        print()
        print(color_text(f"🧠 Brain (cached): {cached}", Color.CYAN))
        print()
        return

    # Show "thinking" indicator
    print()
    print(color_text("🧠 Brain analyzing...", Color.CYAN), end="", flush=True)

    fix = ask_llm_for_fix(error_text)

    # Clear the "analyzing" line
    print("\r" + " " * 30 + "\r", end="")

    print(color_text(f"🧠 Brain: {fix}", Color.CYAN))
    print()

    # Cache for next time
    cache_fix(error_text, fix)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TerminalBrain — wrap a command and analyze its errors with a local LLM"
    )
    parser.add_argument(
        "command",
        nargs="+",
        help="The command to run (e.g. 'python script.py' or 'ls /tmp')"
    )
    args = parser.parse_args()

    # Check ollama
    ok, msg = check_ollama()
    if not ok:
        print(color_text(f"ERROR: {msg}", Color.RED))
        sys.exit(1)

    print(color_text("✓ ollama ready", Color.GREEN))
    print(color_text(f"  Model: {MODEL}", Color.DIM))
    print(color_text(f"  Platform: {platform.system()}", Color.DIM))
    print(color_text(f"  pty available: {HAS_PTY}", Color.DIM))

    # Run the wrapped command
    run_with_brain(args.command)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print("TerminalBrain — wrap a command and get AI fix suggestions")
        print()
        print("Usage:")
        print("  python day10_starter.py <command> [args...]")
        print()
        print("Try one of these to test:")
        print("  python day10_starter.py python -c \"import nonexistent_module\"")
        print("  python day10_starter.py ls /nonexistent_directory")
        print("  python day10_starter.py python -c \"print(undefined_var)\"")
        print()
        sys.exit(0)

    main()
