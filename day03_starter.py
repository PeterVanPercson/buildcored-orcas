"""
BUILDCORED ORCAS — Day 03: VolumeKnuckle
==========================================
Track vertical position of your closed fist.
Fist high = volume up. Fist low = volume down.
No buttons, no keyboard — only position.

Hardware concept: Analog-to-Digital Conversion (ADC)
A potentiometer maps physical position to voltage.
Your fist maps physical position to a digital value.
Same logic. Same math.

YOUR TASK:
1. Map fist height to volume percentage (line marked TODO #1)
2. Add dead zones at top and bottom (line marked TODO #2)
3. Run it: python day03_starter.py
4. Push to GitHub before midnight

CONTROLS:
- Move fist UP → volume increases
- Move fist DOWN → volume decreases
- Press 'q' → quit
"""

import cv2
import mediapipe as mp
import numpy as np
import platform
import subprocess
import sys

# ============================================================
# CROSS-PLATFORM VOLUME CONTROL
# You don't need to change this section.
# It auto-detects your OS and uses the right method.
# ============================================================

OS = platform.system()


def set_system_volume(percent):
    """Set system volume to a percentage (0-100). Works on Mac, Windows, Linux."""
    percent = max(0, min(100, int(percent)))

    try:
        if OS == "Darwin":  # macOS
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {percent}"],
                capture_output=True, timeout=2
            )
        elif OS == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(percent / 100.0, None)
            except ImportError:
                pass  # Display-only mode if pycaw not installed
        else:  # Linux
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                capture_output=True, timeout=2
            )
            if result.returncode != 0:
                subprocess.run(
                    ["amixer", "set", "Master", f"{percent}%"],
                    capture_output=True, timeout=2
                )
    except Exception:
        pass


# ============================================================
# SETUP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

WRIST = 0
current_volume = 50
smoothed_volume = 50.0
SMOOTHING = 0.3


# ============================================================
# TODO #1: Set your dead zones
# ============================================================
# Dead zones are like mechanical stops on a physical knob.
# In the top zone → volume locked at 100%
# In the bottom zone → volume locked at 0%
#
# Without dead zones your hand shakes make it impossible
# to hold exactly 0% or 100%.
#
# Try 0.10 (10%) on each end. Adjust if needed.
#
DEAD_ZONE_TOP = 0.10      # Top 10% = max volume
DEAD_ZONE_BOTTOM = 0.90   # Bottom 10% = min volume


# ============================================================
# TODO #2: Mapping function — the ADC concept
# ============================================================
# This is the core hardware concept.
# A potentiometer converts physical position → voltage → number.
# Your code converts fist position → pixel coordinate → volume.
#
# np.interp does linear mapping:
# np.interp(input, [input_min, input_max], [output_min, output_max])
#
# Remember: y=0 is TOP of frame, y=1 is BOTTOM.
# Top should be loud, bottom should be quiet.
#

def fist_to_volume(y_normalized):
    """Convert fist y-position (0=top, 1=bottom) to volume (0-100)."""

    # Apply dead zones first
    if y_normalized < DEAD_ZONE_TOP:
        return 100.0
    elif y_normalized > DEAD_ZONE_BOTTOM:
        return 0.0

    # Map the remaining range linearly to 0-100
    # Top of active zone → 100, Bottom of active zone → 0
    volume = np.interp(
        y_normalized,
        [DEAD_ZONE_TOP, DEAD_ZONE_BOTTOM],
        [100, 0]
    )
    return volume


# ============================================================
# MAIN LOOP — Sensor → Process → Output
# ============================================================
print("\nVolumeKnuckle is running!")
print(f"OS: {OS}")
print(f"Dead zones: top {DEAD_ZONE_TOP*100:.0f}%, bottom {(1-DEAD_ZONE_BOTTOM)*100:.0f}%")
print("Fist UP = louder. Fist DOWN = quieter.")
print("Show open hand first so MediaPipe can detect, then close fist.")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get wrist y-position (0.0 = top, 1.0 = bottom)
        fist_y = hand_landmarks.landmark[WRIST].y

        # Map to volume
        raw_volume = fist_to_volume(fist_y)

        # Smooth it (prevents jitter)
        smoothed_volume = smoothed_volume + SMOOTHING * (raw_volume - smoothed_volume)
        current_volume = int(smoothed_volume)

        # Actually change system volume
        set_system_volume(current_volume)

        # ---- VISUAL: Volume bar on right side ----
        bar_x = FRAME_W - 60
        bar_top = 50
        bar_bottom = FRAME_H - 50
        bar_height = bar_bottom - bar_top

        # Gray background
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 30, bar_bottom),
                      (50, 50, 50), -1)

        # Colored fill
        fill_height = int(bar_height * current_volume / 100)
        fill_top = bar_bottom - fill_height
        if current_volume < 33:
            bar_color = (0, 200, 0)
        elif current_volume < 66:
            bar_color = (0, 220, 220)
        else:
            bar_color = (0, 80, 255)

        cv2.rectangle(frame, (bar_x, fill_top), (bar_x + 30, bar_bottom),
                      bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 30, bar_bottom),
                      (255, 255, 255), 2)

        # Dead zone lines (dashed appearance)
        dz_top_px = bar_top + int(bar_height * DEAD_ZONE_TOP)
        dz_bot_px = bar_top + int(bar_height * DEAD_ZONE_BOTTOM)
        cv2.line(frame, (bar_x - 5, dz_top_px), (bar_x + 35, dz_top_px), (100, 100, 255), 1)
        cv2.line(frame, (bar_x - 5, dz_bot_px), (bar_x + 35, dz_bot_px), (100, 100, 255), 1)

        # Fist position marker
        fist_px_y = int(fist_y * FRAME_H)
        cv2.circle(frame, (bar_x + 15, fist_px_y), 8, (255, 255, 255), -1)

        # Text
        cv2.putText(frame, f"Volume: {current_volume}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, bar_color, 3)
        cv2.putText(frame, f"Fist Y: {fist_y:.2f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    else:
        cv2.putText(frame, "No hand — show open hand first",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Volume: {current_volume}%", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

    cv2.imshow("VolumeKnuckle - Day 03", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nVolumeKnuckle ended. Final volume: {current_volume}%")
print("See you tomorrow for Day 04!")
