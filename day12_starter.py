"""
BUILDCORED ORCAS — Day 12: SnapAnnotator
==========================================
Press a key to capture a webcam frame.
Send it to a local vision-language model.
Get an object list and ask follow-up questions.

Hardware concept: Vision Inference Pipeline
SENSOR → ISP → MEMORY → INFERENCE → ACTION
Same pipeline runs on Jetson Nano, RPi + Coral,
and every embedded vision system. You're building
the software version on your laptop.

YOUR TASK:
1. Tune the VLM prompt for structured output (TODO #1)
2. Parse the model response into a clean object list (TODO #2)
3. Run it: python day12_starter.py
4. Push to GitHub before midnight

PREREQUISITES:
- ollama running: ollama serve
- Vision model pulled: ollama pull moondream

CONTROLS:
- SPACE → capture frame and analyze
- 1-9   → ask follow-up about object N from last capture
- q     → quit
"""

import cv2
import subprocess
import base64
import tempfile
import os
import sys
import time

# ============================================================
# CHECK OLLAMA + MOONDREAM
# ============================================================

def check_setup():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama not running. Run: ollama serve")
            sys.exit(1)
        if "moondream" not in result.stdout.lower():
            print("ERROR: moondream model not found.")
            print("Fix: ollama pull moondream  (~800 MB)")
            sys.exit(1)
        print("✓ moondream ready")
    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        sys.exit(1)

check_setup()

MODEL = "moondream"
MAX_IMAGE_SIZE = 512  # Resize frames before sending — critical for speed


# ============================================================
# VLM QUERY
# ============================================================

def query_vlm(image_path, prompt):
    """Send an image + prompt to moondream and get a response."""
    try:
        # ollama run with image requires passing the image inline
        result = subprocess.run(
            ["ollama", "run", MODEL, f"{prompt} {image_path}"],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Model timed out]"
    except Exception as e:
        return f"[Error: {e}]"


# ============================================================
# TODO #1: VLM prompt
# ============================================================
# The prompt controls what kind of answer you get.
# Vague prompt → vague answer. Structured prompt → parseable list.
#
# Try to get the model to return a numbered list:
#   1. laptop
#   2. coffee mug
#   3. notebook
#
# So we can show students "1-9 = ask about object N".
#

DESCRIBE_PROMPT = (
    "List the main objects visible in this image. "
    "Respond with a simple numbered list. "
    "One object per line, max 5 objects. "
    "Format: 1. object name"
)


def resize_and_save(frame):
    """Resize frame to MAX_IMAGE_SIZE and save to temp file."""
    h, w = frame.shape[:2]
    scale = MAX_IMAGE_SIZE / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp.name, frame)
    return temp.name


# ============================================================
# TODO #2: Parse model response into object list
# ============================================================
# The model returns text like:
#   "1. laptop\n2. coffee mug\n3. notebook"
# Parse it into: ["laptop", "coffee mug", "notebook"]
#
# Handle edge cases: missing numbers, extra whitespace,
# bullets instead of numbers, etc.
#

def parse_object_list(text):
    """Extract object names from a numbered list."""
    objects = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading number/bullet/dash
        for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
                       "-", "*", "•"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            objects.append(line)
    return objects[:9]  # Max 9 for number-key lookup


# ============================================================
# MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

print("\n" + "=" * 50)
print("  📸 SnapAnnotator")
print("  SPACE = capture | 1-9 = ask about object | q = quit")
print("=" * 50 + "\n")

last_objects = []
last_description = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Show last result on screen
    if last_description:
        for i, obj in enumerate(last_objects):
            cv2.putText(display, f"{i+1}. {obj}", (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(display, "SPACE=snap  1-9=ask  q=quit",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("SnapAnnotator - Day 12", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        print("\n📸 Capturing frame...")
        img_path = resize_and_save(frame)

        print("⏳ Asking the vision model...")
        start = time.time()
        response = query_vlm(img_path, DESCRIBE_PROMPT)
        elapsed = time.time() - start

        os.unlink(img_path)

        print(f"⚡ Response in {elapsed:.1f}s:\n")
        print(response)
        print()

        last_description = response
        last_objects = parse_object_list(response)

        if last_objects:
            print("📋 Parsed objects:")
            for i, obj in enumerate(last_objects):
                print(f"  {i+1}. {obj}")
            print("\nPress 1-9 to ask a follow-up about an object.\n")

    elif ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(last_objects):
            obj = last_objects[idx]
            print(f"\n🔍 Asking about: {obj}")

            # Re-capture current frame for context
            img_path = resize_and_save(frame)
            followup_prompt = f"Tell me about the {obj} in this image in 2 sentences."

            print("⏳ Thinking...")
            start = time.time()
            answer = query_vlm(img_path, followup_prompt)
            elapsed = time.time() - start

            os.unlink(img_path)

            print(f"⚡ ({elapsed:.1f}s)\n{answer}\n")
        else:
            print(f"No object at index {idx+1}")

cap.release()
cv2.destroyAllWindows()
print("\nSnapAnnotator ended. See you tomorrow for Day 13!")
