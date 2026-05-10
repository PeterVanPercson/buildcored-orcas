"""
BUILDCORED ORCAS — Day 29: SilentAssistant
End-to-end pipeline: webcam → VLM → LLM → TTS.
Zero keyboard input. All stages run concurrently.

Hardware concept: Sensor Fusion Pipeline + Latency Profiling
Measure every stage. Find the bottleneck. Optimize it.
Same process as profiling ISR timing on real firmware.

PIPELINE:
  Stage 1: CAPTURE  → Webcam grabs mouth region frame
  Stage 2: VISION   → moondream describes lip movement
  Stage 3: GENERATE → qwen2.5 responds to description
  Stage 4: SPEAK    → pyttsx3 converts response to speech

Stages connect via thread-safe queues.
Each stage runs independently — no blocking between stages.

YOUR TASK:
1. Find the bottleneck stage (TODO #1)
2. Optimize it by 20% (TODO #2)
3. Run: python day29_starter.py

PREREQS:
- ollama running with moondream + qwen2.5:3b
- pip install pyttsx3 (Linux: sudo apt install espeak first)

CONTROLS:
- 's' + Enter → trigger one pipeline cycle manually
- 'q' + Enter → quit
- Or set AUTO_CAPTURE=True for continuous mode
"""

import cv2
import subprocess
import sys
import os
import time
import threading
import queue
import tempfile
import platform

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠ mediapipe not found — using full frame instead of mouth crop")

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    print("⚠ pyttsx3 not found — TTS disabled")
    print("  Mac/Win: pip install pyttsx3")
    print("  Linux:   sudo apt install espeak && pip install pyttsx3")

# ============================================================
# CONFIGURATION
# ============================================================

VISION_MODEL = "moondream"
LLM_MODEL = "qwen2.5:3b"
CAPTURE_INTERVAL = 5.0    # Seconds between auto-captures
AUTO_CAPTURE = False       # Set True for continuous mode
MAX_QUEUE_SIZE = 2         # Drop frames if pipeline backs up
MOUTH_CROP_SIZE = 256      # Resize mouth region before VLM

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    try:
        r = subprocess.run(["ollama", "list"],
                          capture_output=True, text=True, timeout=5)
        models = r.stdout.lower()
        missing = []
        if "moondream" not in models:
            missing.append("moondream")
        if "qwen2.5" not in models:
            missing.append("qwen2.5:3b")
        if missing:
            print(f"Missing models: {', '.join(missing)}")
            print(f"Fix: " + " && ".join(f"ollama pull {m}" for m in missing))
            sys.exit(1)
        print("✓ ollama models ready")
    except:
        print("ERROR: ollama not running. Run: ollama serve")
        sys.exit(1)

check_ollama()


# ============================================================
# LATENCY TRACKER
# ============================================================

class LatencyTracker:
    """Track per-stage timing statistics."""
    def __init__(self):
        self.stages = {
            "capture": [],
            "vision":  [],
            "generate":[],
            "speak":   [],
            "total":   [],
        }
        self.lock = threading.Lock()

    def record(self, stage, duration_ms):
        with self.lock:
            self.stages[stage].append(duration_ms)
            # Keep last 20 measurements
            self.stages[stage] = self.stages[stage][-20:]

    def avg(self, stage):
        with self.lock:
            vals = self.stages[stage]
            return sum(vals) / len(vals) if vals else 0.0

    def report(self):
        print("\n" + "=" * 55)
        print("  📊 Pipeline Latency Report")
        print("=" * 55)
        total = 0
        for stage in ["capture", "vision", "generate", "speak"]:
            avg = self.avg(stage)
            total += avg
            bar = "█" * int(avg / 200) if avg > 0 else ""
            print(f"  {stage:<10} {avg:6.0f}ms  {bar}")
        print(f"  {'TOTAL':<10} {total:6.0f}ms")
        print("=" * 55)

        # TODO #1: Find the bottleneck
        # Which stage has the highest average latency?
        # Print it here and explain why in your README.
        bottleneck = max(
            ["capture", "vision", "generate", "speak"],
            key=self.avg
        )
        print(f"\n  🔴 Bottleneck: {bottleneck} ({self.avg(bottleneck):.0f}ms avg)")
        print(f"  TODO #2: Optimize {bottleneck} by 20%")
        print()


latency = LatencyTracker()


# ============================================================
# PIPELINE QUEUES
# ============================================================

# Each queue connects two adjacent pipeline stages
capture_queue  = queue.Queue(maxsize=MAX_QUEUE_SIZE)  # frames
vision_queue   = queue.Queue(maxsize=MAX_QUEUE_SIZE)  # vision descriptions
generate_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)  # LLM responses

pipeline_running = threading.Event()
pipeline_running.set()


# ============================================================
# STAGE 1: CAPTURE
# ============================================================

def capture_stage(cap):
    """
    Stage 1: Grab webcam frame, crop mouth region, push to queue.
    This is the sensor stage — equivalent to ADC + ISR in firmware.
    """
    if HAS_MEDIAPIPE:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    cycle = 0

    while pipeline_running.is_set():
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        # Crop mouth region
        if HAS_MEDIAPIPE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                # Mouth landmark indices
                mouth_pts = [61, 146, 91, 181, 84, 17, 314, 405,
                             321, 375, 291, 308, 324, 318, 402, 317,
                             14, 87, 178, 88, 95]
                xs = [int(lms[i].x * w) for i in mouth_pts]
                ys = [int(lms[i].y * h) for i in mouth_pts]

                margin = 30
                x1 = max(0, min(xs) - margin)
                y1 = max(0, min(ys) - margin)
                x2 = min(w, max(xs) + margin)
                y2 = min(h, max(ys) + margin)

                mouth_crop = frame[y1:y2, x1:x2]
                if mouth_crop.size > 0:
                    mouth_crop = cv2.resize(
                        mouth_crop, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE)
                    )
                else:
                    mouth_crop = cv2.resize(frame, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))
            else:
                mouth_crop = cv2.resize(frame, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))
        else:
            # No mediapipe — use center crop
            h, w = frame.shape[:2]
            cy, cx = h // 2 + h // 8, w // 2  # Slightly below center
            sz = min(h, w) // 4
            mouth_crop = frame[cy-sz:cy+sz, cx-sz:cx+sz]
            mouth_crop = cv2.resize(mouth_crop, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))

        elapsed_ms = (time.time() - t_start) * 1000
        latency.record("capture", elapsed_ms)

        # Push to queue (drop if full — don't block)
        try:
            capture_queue.put_nowait((frame, mouth_crop, time.time()))
            cycle += 1
        except queue.Full:
            pass  # Drop frame — pipeline backed up

        # Wait before next capture
        if AUTO_CAPTURE:
            time.sleep(max(0, CAPTURE_INTERVAL - elapsed_ms / 1000))
        else:
            time.sleep(0.1)  # Manual mode — just keep webcam alive


# ============================================================
# STAGE 2: VISION
# ============================================================

def vision_stage():
    """
    Stage 2: Send mouth crop to moondream. Get text description.
    Equivalent to embedded vision processor (Jetson, Coral).
    """
    while pipeline_running.is_set():
        try:
            frame, mouth_crop, t_capture = capture_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t_start = time.time()

        # Save crop to temp file
        temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp.name, mouth_crop)

        # Query moondream
        prompt = (
            "Describe what you see in this image of a person's mouth area. "
            "Is the mouth open or closed? Are the lips moving? "
            "Respond in one sentence."
        )
        try:
            result = subprocess.run(
                ["ollama", "run", VISION_MODEL,
                 f"{prompt}\nImage: {temp.name}"],
                capture_output=True, text=True, timeout=30
            )
            description = result.stdout.strip()
        except subprocess.TimeoutExpired:
            description = "mouth region captured, vision model timed out"
        except Exception as e:
            description = f"capture complete ({e})"
        finally:
            try:
                os.unlink(temp.name)
            except:
                pass

        elapsed_ms = (time.time() - t_start) * 1000
        latency.record("vision", elapsed_ms)

        print(f"\n[Vision {elapsed_ms:.0f}ms] {description[:80]}...")

        # Push to next stage
        try:
            vision_queue.put_nowait((description, t_capture, time.time()))
        except queue.Full:
            pass


# ============================================================
# STAGE 3: GENERATE
# ============================================================

def generate_stage():
    """
    Stage 3: LLM generates response to vision description.
    Equivalent to MCU processing sensor data.
    """
    while pipeline_running.is_set():
        try:
            description, t_capture, t_vision = vision_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t_start = time.time()

        prompt = (
            f"A camera sees: '{description}'\n"
            f"You are a silent assistant. "
            f"Respond to what the person might be trying to communicate. "
            f"Reply in exactly one short sentence."
        )

        try:
            result = subprocess.run(
                ["ollama", "run", LLM_MODEL, prompt],
                capture_output=True, text=True, timeout=30
            )
            response = result.stdout.strip()
        except subprocess.TimeoutExpired:
            response = "Processing took too long. Try again."
        except Exception as e:
            response = f"Pipeline error: {e}"

        elapsed_ms = (time.time() - t_start) * 1000
        latency.record("generate", elapsed_ms)

        print(f"\n[Generate {elapsed_ms:.0f}ms] {response}")

        # Push to TTS stage
        try:
            generate_queue.put_nowait((response, t_capture, time.time()))
        except queue.Full:
            pass


# ============================================================
# STAGE 4: SPEAK
# ============================================================

def speak_stage():
    """
    Stage 4: Convert LLM response to speech.
    Equivalent to DAC + speaker driver output.
    """
    engine = None
    if HAS_TTS:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            print("✓ TTS engine ready")
        except Exception as e:
            print(f"⚠ TTS init failed: {e}")
            engine = None

    while pipeline_running.is_set():
        try:
            response, t_capture, t_generate = generate_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        t_start = time.time()

        if engine:
            try:
                engine.say(response)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            print(f"\n[TTS disabled] Would say: '{response}'")
            time.sleep(0.5)

        elapsed_ms = (time.time() - t_start) * 1000
        latency.record("speak", elapsed_ms)

        # Total pipeline latency
        total_ms = (time.time() - t_capture) * 1000
        latency.record("total", total_ms)

        print(f"\n[Speak {elapsed_ms:.0f}ms] [Total pipeline: {total_ms:.0f}ms]")

        # TODO #2: Optimize the bottleneck stage
        # After finding the bottleneck from latency.report(),
        # try ONE of these optimizations:
        #
        # If VISION is bottleneck:
        #   - Reduce MOUTH_CROP_SIZE from 256 to 128
        #   - Cache identical-looking frames (hash-based)
        #   - Run vision every 2nd cycle only
        #
        # If GENERATE is bottleneck:
        #   - Use a shorter/tighter prompt
        #   - Cache common responses
        #   - Switch to a smaller model if available
        #
        # If SPEAK is bottleneck:
        #   - Reduce engine.setProperty("rate") to speak faster
        #   - Use subprocess espeak directly (faster startup)
        #   - Skip TTS for repeated identical responses


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 55)
    print("  🤫 SilentAssistant — Day 29 (EXPERT)")
    print("=" * 55)
    print()
    print("  Pipeline: Webcam → VLM → LLM → TTS")
    print(f"  Vision model: {VISION_MODEL}")
    print(f"  LLM model:    {LLM_MODEL}")
    print(f"  TTS:          {'pyttsx3' if HAS_TTS else 'disabled'}")
    print(f"  Auto-capture: {AUTO_CAPTURE}")
    print()
    print("  Commands:")
    print("  's' + Enter → trigger one pipeline cycle")
    print("  'r' + Enter → show latency report")
    print("  'q' + Enter → quit")
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: No webcam found.")
        sys.exit(1)

    print("✓ Webcam open")
    print()

    # Start pipeline threads
    threads = [
        threading.Thread(target=capture_stage, args=(cap,), daemon=True),
        threading.Thread(target=vision_stage, daemon=True),
        threading.Thread(target=generate_stage, daemon=True),
        threading.Thread(target=speak_stage, daemon=True),
    ]

    for t in threads:
        t.start()

    print("✓ Pipeline running\n")

    # Webcam display thread
    def show_webcam():
        while pipeline_running.is_set():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "SilentAssistant D29",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Auto: {'ON' if AUTO_CAPTURE else 'OFF'}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("SilentAssistant - Day 29", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pipeline_running.clear()
                break

    webcam_thread = threading.Thread(target=show_webcam, daemon=True)
    webcam_thread.start()

    # Command loop
    try:
        while pipeline_running.is_set():
            try:
                cmd = input().strip().lower()
            except EOFError:
                break

            if cmd == 'q':
                break
            elif cmd == 's':
                print("[Manual trigger] Capturing...")
                if AUTO_CAPTURE:
                    print("Already in auto mode")
                else:
                    # Force one capture by temporarily enabling
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        h, w = frame.shape[:2]
                        crop = cv2.resize(frame, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))
                        try:
                            capture_queue.put_nowait((frame, crop, time.time()))
                            print("✓ Frame queued for processing")
                        except queue.Full:
                            print("Queue full — pipeline still processing")
            elif cmd == 'r':
                latency.report()

    except KeyboardInterrupt:
        pass
    finally:
        pipeline_running.clear()
        cap.release()
        cv2.destroyAllWindows()

    latency.report()
    print("\nSilentAssistant ended. One more day to go! 🐋")


if __name__ == "__main__":
    main()
