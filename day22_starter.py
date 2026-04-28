"""
BUILDCORED ORCAS — Day 22: CircuitWhisperer
Photograph a circuit schematic. Local vision model
identifies components and describes the circuit.

Hardware concept: Schematic Reading
Every hardware engineer reads schematics. This builds
your visual vocabulary for hardware documentation.

YOUR TASK:
1. Tune the component detection prompt (TODO #1)
2. Add a wiring error detection query (TODO #2)
3. Run: python day22_starter.py

PREREQS:
- ollama running: ollama serve
- Vision model: ollama pull moondream

CONTROLS:
- SPACE → capture from webcam
- 'f'   → load from file (place circuit.jpg/png in folder)
- 't'   → use generated test schematic
- 'q'   → quit
"""

import cv2
import subprocess
import sys
import os
import time
import tempfile
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("pip install Pillow for best results")

# ============================================================
# CHECK OLLAMA + MOONDREAM
# ============================================================

def check_setup():
    try:
        r = subprocess.run(["ollama", "list"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            print("ERROR: ollama not running. Run: ollama serve")
            sys.exit(1)
        if "moondream" not in r.stdout.lower():
            print("ERROR: moondream not found.")
            print("Fix: ollama pull moondream")
            sys.exit(1)
        print("✓ moondream ready")
    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        sys.exit(1)

check_setup()

MODEL = "moondream"
MAX_SIZE = 512


# ============================================================
# GENERATE TEST SCHEMATIC
# ============================================================

def generate_test_circuit(output_path="test_circuit.png"):
    """
    Draw a simple RC low-pass filter schematic.
    Used as fallback when no real circuit photo is available.
    White background, black lines — maximizes model accuracy.
    """
    if not HAS_PIL:
        # Fallback: draw with OpenCV
        img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Draw a simple rectangle as "circuit"
        cv2.rectangle(img, (50, 100), (200, 200), (0, 0, 0), 2)
        cv2.rectangle(img, (250, 100), (400, 200), (0, 0, 0), 2)
        cv2.line(img, (200, 150), (250, 150), (0, 0, 0), 2)
        cv2.putText(img, "R1", (100, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "C1", (300, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "RC Low-pass Filter", (100, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(output_path, img)
        return output_path

    W, H = 600, 350
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    lw = 3

    # Title
    draw.text((W//2 - 100, 10), "RC Low-Pass Filter", fill="black")

    # ---- VIN node ----
    draw.text((20, 145), "VIN", fill="black")
    draw.line([(70, 150), (120, 150)], fill="black", width=lw)

    # ---- Resistor R1 (zig-zag) ----
    rx, ry = 120, 150
    draw.text((145, 115), "R1 = 10kΩ", fill="black")
    zag_pts = [(rx, ry)]
    for i in range(8):
        offset = 8 if i % 2 == 0 else -8
        zag_pts.append((rx + 20 + i * 20, ry + offset))
    zag_pts.append((rx + 200, ry))
    draw.line(zag_pts, fill="black", width=lw)

    # ---- Node after R1 ----
    nx, ny = 320, 150
    draw.line([(nx, ny), (nx + 60, ny)], fill="black", width=lw)

    # ---- Capacitor C1 (two parallel plates) ----
    cx, cy = 380, 150
    draw.text((370, 115), "C1 = 100nF", fill="black")
    draw.line([(cx, cy), (cx, cy - 40)], fill="black", width=lw)  # top plate
    draw.line([(cx - 30, cy - 40), (cx + 30, cy - 40)], fill="black", width=lw)
    draw.line([(cx - 30, cy - 52), (cx + 30, cy - 52)], fill="black", width=lw)
    draw.line([(cx, cy - 52), (cx, cy - 80)], fill="black", width=lw)  # bottom plate

    # Ground from cap bottom
    draw.line([(cx, cy + 0), (cx, cy + 60)], fill="black", width=lw)
    draw.line([(cx - 30, cy + 60), (cx + 30, cy + 60)], fill="black", width=lw)
    draw.line([(cx - 18, cy + 72), (cx + 18, cy + 72)], fill="black", width=lw)
    draw.line([(cx - 6, cy + 84), (cx + 6, cy + 84)], fill="black", width=lw)
    draw.text((cx + 5, cy + 60), "GND", fill="black")

    # Ground from VIN
    draw.line([(70, 200), (70, 260)], fill="black", width=lw)
    draw.line([(40, 260), (100, 260)], fill="black", width=lw)
    draw.line([(52, 272), (88, 272)], fill="black", width=lw)
    draw.line([(62, 284), (78, 284)], fill="black", width=lw)
    draw.text((75, 260), "GND", fill="black")

    # VIN vertical line
    draw.line([(70, 150), (70, 200)], fill="black", width=lw)

    # VOUT label
    draw.line([(nx + 60, ny), (nx + 100, ny)], fill="black", width=lw)
    draw.text((nx + 100, 135), "VOUT", fill="black")

    # Dot at junction
    draw.ellipse([(nx - 5, ny - 5), (nx + 5, ny + 5)], fill="black")

    img.save(output_path)
    print(f"✓ Generated test schematic: {output_path}")
    return output_path


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img_path, max_size=MAX_SIZE):
    """Resize and enhance contrast for better model accuracy."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Enhance contrast (helps with hand-drawn circuits in poor lighting)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp.name, enhanced)
    return temp.name


# ============================================================
# VLM QUERIES
# ============================================================

# TODO #1: Component detection prompt
# Getting structured output from moondream is hard.
# Try these variations:
#
# Option A: Strict numbered list
COMPONENT_PROMPT = (
    "List every electronic component visible in this circuit schematic. "
    "Respond with a simple numbered list, one component per line. "
    "Include: resistors, capacitors, inductors, diodes, transistors, "
    "op-amps, voltage sources, ground symbols. "
    "If you see none, say 'No components identified'. "
    "Maximum 8 items. Format:\n1. component name"
)

# Option B: More structured (try this if A gives bad results)
COMPONENT_PROMPT_V2 = (
    "Analyze this electronic circuit. "
    "Return only a JSON object with these fields: "
    '{"components": ["list of component names"], '
    '"circuit_type": "what this circuit does", '
    '"confidence": "high/medium/low"}. '
    "No explanation, just the JSON."
)

FUNCTION_PROMPT = (
    "What does this electronic circuit do? "
    "Describe its function in 2 sentences. "
    "Be specific about signal flow."
)

# TODO #2: Wiring error detection
# Add a prompt that asks the model to identify potential
# wiring errors or unusual connections.
# Examples: missing ground, floating inputs, short circuits,
# incorrect polarities, etc.
#
WIRING_ERROR_PROMPT = (
    "Look at this circuit schematic carefully. "
    "Are there any obvious wiring errors or potential problems? "
    "List up to 3 specific issues, or say 'No obvious errors detected'. "
    "Be concise."
)


def query_vlm(image_path, prompt):
    """Send image + prompt to moondream via ollama."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL,
             f"{prompt}\n\nImage: {image_path}"],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Model timed out — try a smaller image]"
    except Exception as e:
        return f"[Error: {e}]"


def analyze_circuit(image_path):
    """Run all three analyses on a circuit image."""
    print("\n" + "─" * 50)
    print("🔍 Analyzing circuit...")
    print("─" * 50)

    # Preprocess
    processed = preprocess_image(image_path)
    if processed is None:
        print("ERROR: Could not load image.")
        return
    path = processed

    # 1. Component detection
    print("\n[1/3] Detecting components...")
    start = time.time()
    components = query_vlm(path, COMPONENT_PROMPT)
    print(f"⚡ ({time.time()-start:.1f}s)")
    print(f"\n{components}\n")

    # 2. Circuit function
    print("[2/3] Identifying circuit function...")
    start = time.time()
    function = query_vlm(path, FUNCTION_PROMPT)
    print(f"⚡ ({time.time()-start:.1f}s)")
    print(f"\n{function}\n")

    # 3. Wiring errors
    print("[3/3] Checking for wiring errors...")
    start = time.time()
    errors = query_vlm(path, WIRING_ERROR_PROMPT)
    print(f"⚡ ({time.time()-start:.1f}s)")
    print(f"\n{errors}\n")

    print("─" * 50)
    print("Analysis complete.")

    # Cleanup
    try:
        os.unlink(processed)
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 55)
    print("  ⚡ CircuitWhisperer — Day 22")
    print("=" * 55)
    print()
    print("  SPACE = capture from webcam")
    print("  'f'   = load circuit.jpg/circuit.png from folder")
    print("  't'   = use generated test schematic")
    print("  'q'   = quit")
    print()

    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    has_webcam = cap.isOpened()

    if not has_webcam:
        print("⚠️  No webcam found. Using 't' for test schematic.")

    while True:
        if has_webcam:
            ret, frame = cap.read()
            if ret:
                display = frame.copy()
                cv2.putText(display, "SPACE=capture  f=file  t=test  q=quit",
                            (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("CircuitWhisperer - Day 22", display)

        key = cv2.waitKey(1) & 0xFF if has_webcam else ord(input("\nCommand (t/f/q): ").strip())

        if key == ord('q'):
            break

        elif key == ord(' ') and has_webcam:
            # Capture from webcam
            temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp.name, frame)
            print(f"\n📸 Captured frame → {temp.name}")
            analyze_circuit(temp.name)
            os.unlink(temp.name)

        elif key == ord('f'):
            # Load from file
            for fname in ["circuit.jpg", "circuit.png",
                          "circuit.jpeg", "schematic.jpg"]:
                if os.path.exists(fname):
                    print(f"\n📂 Loading: {fname}")
                    analyze_circuit(fname)
                    break
            else:
                print("⚠️  No circuit image found in current folder.")
                print("   Place a circuit photo named 'circuit.jpg' here.")

        elif key == ord('t'):
            # Generate test schematic
            print("\n🔧 Generating RC low-pass filter schematic...")
            test_path = generate_test_circuit()
            analyze_circuit(test_path)

    if has_webcam:
        cap.release()
    cv2.destroyAllWindows()
    print("\nCircuitWhisperer ended. See you tomorrow for Day 23!")


if __name__ == "__main__":
    main()
