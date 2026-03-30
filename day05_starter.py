"""
BUILDCORED ORCAS — Day 05: FaceEQ
===================================
Use head yaw (left/right) to scrub through an audio track.
Use head pitch (up/down) to change playback speed.
Your head becomes a DJ controller.

Hardware concept: Rotary Encoder
A rotary encoder outputs angular displacement as digital
pulses. Your head rotation is the encoder — yaw and pitch
are two independent axes feeding position data.

YOUR TASK:
1. Tune the yaw-to-scrub mapping (TODO #1)
2. Tune the pitch-to-speed mapping (TODO #2)
3. Add a visual head position indicator (TODO #3)
4. Drop a .mp3 file named "track.mp3" in this folder
5. Run it: python day05_starter.py
6. Push to GitHub before midnight

CONTROLS:
- Turn head LEFT → rewind
- Turn head RIGHT → fast forward
- Tilt head UP → speed up playback
- Tilt head DOWN → slow down playback
- Press SPACE → play/pause
- Press 'r' → reset to start
- Press 'q' → quit
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import os

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize audio
pygame.mixer.init(frequency=44100)

TRACK_FILE = "track.mp3"
if os.path.exists(TRACK_FILE):
    pygame.mixer.music.load(TRACK_FILE)
    print(f"Loaded: {TRACK_FILE}")
else:
    print(f"ERROR: '{TRACK_FILE}' not found!")
    print("Place any .mp3 file in this folder and name it 'track.mp3'")
    sys.exit(1)

# Head pose estimation landmarks
# We use nose tip and forehead to estimate head angle
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FOREHEAD = 10


def estimate_head_pose(landmarks, w, h):
    """
    Estimate head yaw (left/right) and pitch (up/down)
    from facial landmarks.

    Returns:
        yaw: negative = looking left, positive = looking right
             roughly -30 to +30 degrees
        pitch: negative = looking down, positive = looking up
               roughly -20 to +20 degrees
    """
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    # YAW: compare nose x-position relative to eye midpoint
    eye_mid_x = (left_eye.x + right_eye.x) / 2
    eye_distance = abs(right_eye.x - left_eye.x)

    if eye_distance > 0:
        yaw = (nose.x - eye_mid_x) / eye_distance
        yaw = yaw * 60  # Scale to approximate degrees
    else:
        yaw = 0

    # PITCH: compare nose-to-chin vs forehead-to-nose ratio
    nose_to_chin = chin.y - nose.y
    forehead_to_nose = nose.y - forehead.y

    if forehead_to_nose > 0:
        pitch_ratio = nose_to_chin / forehead_to_nose
        pitch = (pitch_ratio - 1.0) * 40  # Scale to approximate degrees
    else:
        pitch = 0

    return yaw, pitch


# ============================================================
# TODO #1: Yaw-to-scrub mapping
# ============================================================
# YAW_DEAD_ZONE: head angle (degrees) where no scrubbing happens.
#   Prevents accidental scrubbing from small head movements.
#   Think of it as the dead zone on a joystick.
#
# SCRUB_SPEED: how many seconds to skip per frame when head
#   is turned. Higher = faster scrubbing.
#
YAW_DEAD_ZONE = 5.0      # degrees — no scrub within this range
SCRUB_SPEED = 0.15        # seconds per frame at max head turn


# ============================================================
# TODO #2: Pitch-to-speed mapping
# ============================================================
# Head tilted UP → faster playback (up to MAX_SPEED)
# Head tilted DOWN → slower playback (down to MIN_SPEED)
# Head level → normal speed (1.0)
#
# PITCH_DEAD_ZONE: degrees where speed stays at 1.0
# MIN_SPEED / MAX_SPEED: limits so audio doesn't go crazy
#
PITCH_DEAD_ZONE = 5.0     # degrees — normal speed within this
MIN_SPEED = 0.5            # slowest playback (half speed)
MAX_SPEED = 2.0            # fastest playback (double speed)


# ============================================================
# STATE
# ============================================================

is_playing = False
track_position = 0.0        # current position in seconds
playback_speed = 1.0         # current speed multiplier
current_yaw = 0.0
current_pitch = 0.0

# Get track length (approximate — pygame doesn't expose this easily)
# We'll estimate from file or default to 180 seconds
TRACK_LENGTH = 180.0  # seconds — adjust if your track is shorter/longer

# Smoothing
smooth_yaw = 0.0
smooth_pitch = 0.0
SMOOTHING = 0.3


# ============================================================
# MAIN LOOP
# ============================================================
print("\nFaceEQ is running!")
print(f"Yaw dead zone: ±{YAW_DEAD_ZONE}°")
print(f"Pitch dead zone: ±{PITCH_DEAD_ZONE}°")
print(f"Speed range: {MIN_SPEED}x to {MAX_SPEED}x")
print("Turn head LEFT/RIGHT to scrub. Tilt UP/DOWN for speed.")
print("SPACE = play/pause, 'r' = reset, 'q' = quit\n")

# Start playing
pygame.mixer.music.play()
is_playing = True
print("▶ Playing")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get head pose
        raw_yaw, raw_pitch = estimate_head_pose(landmarks, FRAME_W, FRAME_H)

        # Smooth values
        smooth_yaw += SMOOTHING * (raw_yaw - smooth_yaw)
        smooth_pitch += SMOOTHING * (raw_pitch - smooth_pitch)
        current_yaw = smooth_yaw
        current_pitch = smooth_pitch

        # ---- SCRUBBING (yaw) ----
        if abs(current_yaw) > YAW_DEAD_ZONE and is_playing:
            # How far past dead zone (0 to ~25 degrees)
            scrub_amount = current_yaw - (YAW_DEAD_ZONE if current_yaw > 0 else -YAW_DEAD_ZONE)
            # Normalize and apply speed
            scrub_seconds = (scrub_amount / 25.0) * SCRUB_SPEED

            track_position += scrub_seconds
            track_position = max(0, min(TRACK_LENGTH, track_position))

            # Jump to new position
            pygame.mixer.music.set_pos(track_position)

        # ---- SPEED (pitch) ----
        if abs(current_pitch) > PITCH_DEAD_ZONE:
            pitch_amount = current_pitch - (PITCH_DEAD_ZONE if current_pitch > 0 else -PITCH_DEAD_ZONE)
            # Map pitch to speed range
            speed_offset = (pitch_amount / 20.0) * (MAX_SPEED - 1.0)
            playback_speed = 1.0 + speed_offset
            playback_speed = max(MIN_SPEED, min(MAX_SPEED, playback_speed))
        else:
            playback_speed = 1.0

        # ---- Draw nose point ----
        nose = landmarks[NOSE_TIP]
        nose_x = int(nose.x * FRAME_W)
        nose_y = int(nose.y * FRAME_H)
        cv2.circle(frame, (nose_x, nose_y), 6, (0, 255, 255), -1)

        # ============================================================
        # TODO #3: Visual head position indicator
        # ============================================================
        # Draw a crosshair or gauge showing current yaw and pitch.
        # This tells the user where their "knob" is currently set.
        #
        # Simple version: a dot on a grid that moves with head angle
        #

        # --- Gauge: circle with dot showing head position ---
        gauge_center_x = FRAME_W - 80
        gauge_center_y = 80
        gauge_radius = 50

        # Background circle
        cv2.circle(frame, (gauge_center_x, gauge_center_y), gauge_radius,
                   (50, 50, 50), -1)
        cv2.circle(frame, (gauge_center_x, gauge_center_y), gauge_radius,
                   (150, 150, 150), 2)

        # Crosshair lines
        cv2.line(frame, (gauge_center_x - gauge_radius, gauge_center_y),
                 (gauge_center_x + gauge_radius, gauge_center_y), (80, 80, 80), 1)
        cv2.line(frame, (gauge_center_x, gauge_center_y - gauge_radius),
                 (gauge_center_x, gauge_center_y + gauge_radius), (80, 80, 80), 1)

        # Dead zone circle
        dz_radius = int(gauge_radius * (YAW_DEAD_ZONE / 30.0))
        cv2.circle(frame, (gauge_center_x, gauge_center_y), dz_radius,
                   (80, 80, 80), 1)

        # Dot showing current head position
        dot_x = int(gauge_center_x + (current_yaw / 30.0) * gauge_radius)
        dot_y = int(gauge_center_y - (current_pitch / 20.0) * gauge_radius)
        dot_x = max(gauge_center_x - gauge_radius, min(gauge_center_x + gauge_radius, dot_x))
        dot_y = max(gauge_center_y - gauge_radius, min(gauge_center_y + gauge_radius, dot_y))

        # Color based on whether we're in dead zone
        in_dead_zone = abs(current_yaw) < YAW_DEAD_ZONE and abs(current_pitch) < PITCH_DEAD_ZONE
        dot_color = (150, 150, 150) if in_dead_zone else (0, 255, 255)
        cv2.circle(frame, (dot_x, dot_y), 6, dot_color, -1)

    else:
        cv2.putText(frame, "No face detected — stay centered",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---- TEXT OVERLAY ----
    status = "PLAYING" if is_playing else "PAUSED"
    status_color = (0, 255, 0) if is_playing else (0, 0, 255)

    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Yaw: {current_yaw:+.1f} deg", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Pitch: {current_pitch:+.1f} deg", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Speed: {playback_speed:.2f}x", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(frame, f"Position: {track_position:.1f}s", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Scrub direction indicator
    if abs(current_yaw) > YAW_DEAD_ZONE:
        direction = ">> FORWARD" if current_yaw > 0 else "<< REWIND"
        cv2.putText(frame, direction, (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "SPACE=play/pause  r=reset  q=quit",
                (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    cv2.imshow("FaceEQ - Day 05", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if is_playing:
            pygame.mixer.music.pause()
            is_playing = False
            print("⏸ Paused")
        else:
            pygame.mixer.music.unpause()
            is_playing = True
            print("▶ Playing")
    elif key == ord('r'):
        track_position = 0.0
        pygame.mixer.music.rewind()
        print("⏮ Reset to start")

    # Update track position estimate
    if is_playing:
        track_position += (1 / 30.0) * playback_speed  # Approximate at 30fps
        track_position = min(track_position, TRACK_LENGTH)

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("\nFaceEQ ended. See you tomorrow for Day 06!")
