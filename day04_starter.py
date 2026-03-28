"""
BUILDCORED ORCAS — Day 04: BlinkLock
======================================
Count rapid blinks using Eye Aspect Ratio (EAR).
Three rapid blinks = lock. Wink or PIN = unlock.

Hardware concept: Debounce + State Machine
Hardware buttons bounce (flicker on/off when pressed).
Firmware uses state machines to filter bounces into
clean single-press events. This is the same pattern.

YOUR TASK:
1. Tune the EAR threshold (line marked TODO #1)
2. Adjust the timing window for blink counting (TODO #2)
3. Understand the state machine transitions (TODO #3)
4. Run it: python day04_starter.py
5. Push to GitHub before midnight

CONTROLS:
- Blink 3 times rapidly → LOCK
- Press 'u' → UNLOCK (PIN fallback)
- Press 'q' → quit

STATE MACHINE:
  IDLE → (blink detected) → COUNTING
  COUNTING → (3 blinks within window) → LOCKED
  COUNTING → (timeout, not enough blinks) → IDLE
  LOCKED → (press 'u') → IDLE
"""

import cv2
import mediapipe as mp
import time
import sys

# ============================================================
# SETUP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe FaceMesh eye landmark indices
# Left eye: top [159, 145], bottom [145, 159] — vertical
# Right eye: similar pattern
# We use specific points that give reliable EAR

# Left eye landmarks (vertical pair + horizontal pair)
LEFT_EYE_TOP = [159, 160, 161]
LEFT_EYE_BOTTOM = [145, 144, 153]
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

# Right eye landmarks
RIGHT_EYE_TOP = [386, 387, 388]
RIGHT_EYE_BOTTOM = [374, 373, 380]
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263


def get_ear(landmarks, top_ids, bottom_ids, left_id, right_id):
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR = (vertical distance) / (horizontal distance)

    When eye is open: EAR ~ 0.25-0.35
    When eye is closed: EAR < 0.20

    This is the same principle as reading a sensor value
    and comparing it to a threshold.
    """
    # Average vertical distance (top to bottom of eye)
    vertical = 0
    for t, b in zip(top_ids, bottom_ids):
        vertical += abs(landmarks[t].y - landmarks[b].y)
    vertical /= len(top_ids)

    # Horizontal distance (corner to corner)
    horizontal = abs(landmarks[left_id].x - landmarks[right_id].x)

    if horizontal == 0:
        return 0.0

    ear = vertical / horizontal
    return ear


# ============================================================
# TODO #1: Tune the EAR threshold
# ============================================================
# When EAR drops BELOW this value, we consider the eye "closed"
# (a blink).
#
# Too low (0.15): only detects very hard squeezes
# Too high (0.30): false triggers from normal squinting
# Start with 0.21 and adjust while watching the EAR value on screen
#
EAR_THRESHOLD = 0.21  # <-- Adjust this


# ============================================================
# TODO #2: Timing configuration
# ============================================================
# BLINK_TIME_WINDOW: how many seconds to count blinks in.
#   3 blinks must happen within this window.
#   Too short (0.5s): impossible to blink 3 times fast enough
#   Too long (5.0s): slow blinks count, feels broken
#
# MIN_BLINK_DURATION: minimum frames an eye must be closed
#   to count as a blink (debounce!). Prevents flicker.
#   This IS the debounce — same as firmware debounce timers.
#
BLINK_TIME_WINDOW = 2.0    # seconds to get 3 blinks
BLINKS_TO_LOCK = 3         # how many blinks to trigger lock
MIN_BLINK_DURATION = 2     # minimum frames eye must be closed


# ============================================================
# TODO #3: State machine
# ============================================================
# Three states — just like a firmware interrupt handler:
#
#   IDLE     → waiting for first blink
#   COUNTING → first blink detected, counting more
#   LOCKED   → 3 blinks detected, screen is locked
#
# Transitions:
#   IDLE → blink → COUNTING (start timer)
#   COUNTING → 3 blinks before timeout → LOCKED
#   COUNTING → timeout → IDLE (reset)
#   LOCKED → 'u' key → IDLE (unlock)
#

STATE_IDLE = "IDLE"
STATE_COUNTING = "COUNTING"
STATE_LOCKED = "LOCKED"

# Current state
state = STATE_IDLE

# Blink tracking variables
blink_count = 0
counting_start_time = 0
eye_closed_frames = 0
eye_was_closed = False


# ============================================================
# MAIN LOOP
# ============================================================
print("\nBlinkLock is running!")
print(f"EAR threshold: {EAR_THRESHOLD}")
print(f"Blink {BLINKS_TO_LOCK}x within {BLINK_TIME_WINDOW}s to LOCK")
print(f"Press 'u' to UNLOCK, 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    # ---- If LOCKED, show lock screen ----
    if state == STATE_LOCKED:
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "LOCKED", (w // 2 - 120, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        cv2.putText(frame, "Press 'u' to unlock", (w // 2 - 130, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("BlinkLock - Day 04", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('u'):
            state = STATE_IDLE
            blink_count = 0
            print("UNLOCKED")
        elif key == ord('q'):
            break
        continue

    # ---- Process face landmarks ----
    current_ear = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR for both eyes and average
        left_ear = get_ear(landmarks,
                           LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                           LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right_ear = get_ear(landmarks,
                            RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                            RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        current_ear = (left_ear + right_ear) / 2

        # ---- BLINK DETECTION (with debounce) ----
        eye_is_closed = current_ear < EAR_THRESHOLD

        if eye_is_closed:
            eye_closed_frames += 1
        else:
            # Eye just opened — was it a real blink?
            # Only count if eye was closed for enough frames (DEBOUNCE)
            if eye_closed_frames >= MIN_BLINK_DURATION:
                # Valid blink detected!

                if state == STATE_IDLE:
                    # First blink — start counting
                    state = STATE_COUNTING
                    blink_count = 1
                    counting_start_time = time.time()
                    print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")

                elif state == STATE_COUNTING:
                    blink_count += 1
                    print(f"Blink {blink_count}/{BLINKS_TO_LOCK}")

                    if blink_count >= BLINKS_TO_LOCK:
                        state = STATE_LOCKED
                        blink_count = 0
                        print("LOCKED! Press 'u' to unlock.")

            eye_closed_frames = 0

        # ---- TIMEOUT CHECK ----
        if state == STATE_COUNTING:
            elapsed = time.time() - counting_start_time
            if elapsed > BLINK_TIME_WINDOW:
                # Took too long — reset
                print(f"Timeout. Only got {blink_count}/{BLINKS_TO_LOCK} blinks. Resetting.")
                state = STATE_IDLE
                blink_count = 0

        # ---- Draw eye landmarks ----
        for idx in LEFT_EYE_TOP + LEFT_EYE_BOTTOM + RIGHT_EYE_TOP + RIGHT_EYE_BOTTOM:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # ---- DISPLAY ----

    # EAR value
    ear_color = (0, 0, 255) if current_ear < EAR_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # State
    state_colors = {
        STATE_IDLE: (200, 200, 200),
        STATE_COUNTING: (0, 200, 255),
        STATE_LOCKED: (0, 0, 255),
    }
    cv2.putText(frame, f"State: {state}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_colors[state], 2)

    # Blink counter (when counting)
    if state == STATE_COUNTING:
        elapsed = time.time() - counting_start_time
        remaining = max(0, BLINK_TIME_WINDOW - elapsed)
        cv2.putText(frame, f"Blinks: {blink_count}/{BLINKS_TO_LOCK}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Visual countdown bar
        bar_width = int((remaining / BLINK_TIME_WINDOW) * 200)
        cv2.rectangle(frame, (10, 170), (10 + bar_width, 180), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, 170), (210, 180), (100, 100, 100), 1)

    # Instructions at bottom
    cv2.putText(frame, f"Blink {BLINKS_TO_LOCK}x rapidly to LOCK | 'u' = unlock | 'q' = quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("BlinkLock - Day 04", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u') and state == STATE_LOCKED:
        state = STATE_IDLE
        blink_count = 0
        print("UNLOCKED")

cap.release()
cv2.destroyAllWindows()
print("\nBlinkLock ended. See you tomorrow for Day 05!")
