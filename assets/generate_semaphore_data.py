#!/usr/bin/env python3
"""
Generate a synthetic semaphore flag landmarks dataset for Day 27 (SignalFlags).

Semaphore flag positions encode letters A-Z using arm angles.
This generates synthetic MediaPipe-style hand landmark data (21 landmarks, x/y/z each)
for each letter, with multiple samples per letter to create a training set.

Each row represents one hand pose (right hand).
The key discriminating features are the wrist-to-fingertip vectors,
which encode the arm angle for that semaphore position.

Output: semaphore_landmarks.csv
"""

import csv
import math
import random
import os

random.seed(42)

# Semaphore flag angles (in degrees from straight down, clockwise)
# Each letter has (left_arm_angle, right_arm_angle)
# 0 = straight down, 45 = down-left, 90 = left, 135 = up-left, 180 = up, etc.
SEMAPHORE_ANGLES = {
    'A': (225, 270),
    'B': (180, 270),
    'C': (135, 270),
    'D': (90, 270),
    'E': (270, 45),
    'F': (270, 90),
    'G': (270, 135),
    'H': (225, 180),
    'I': (225, 90),
    'J': (90, 0),
    'K': (225, 0),
    'L': (225, 45),
    'M': (225, 90),
    'N': (225, 135),
    'O': (180, 0),
    'P': (180, 45),
    'Q': (180, 90),
    'R': (180, 135),
    'S': (135, 0),
    'T': (135, 45),
    'U': (135, 90),
    'V': (90, 0),
    'W': (45, 90),
    'X': (45, 135),
    'Y': (90, 45),
    'Z': (45, 270),
}

# MediaPipe hand has 21 landmarks (indices 0-20)
# We simulate a simplified hand where the key signal is the overall
# hand orientation (wrist to middle fingertip direction)
NUM_LANDMARKS = 21
SAMPLES_PER_LETTER = 40  # 40 samples per letter = 1040 total

def angle_to_direction(angle_deg):
    """Convert semaphore angle to a unit direction vector (x, y)."""
    rad = math.radians(angle_deg)
    return (math.sin(rad), -math.cos(rad))  # y-axis inverted for screen coords

def generate_hand_landmarks(arm_angle_deg, noise_level=0.015):
    """
    Generate 21 synthetic hand landmarks for a given arm angle.
    Landmarks roughly follow MediaPipe's hand model.
    The overall hand orientation encodes the semaphore arm position.
    """
    dx, dy = angle_to_direction(arm_angle_deg)

    # Base wrist position (centered around 0.5, 0.7 in normalized coords)
    wrist_x = 0.5 + random.gauss(0, 0.02)
    wrist_y = 0.7 + random.gauss(0, 0.02)

    landmarks = []

    # Landmark layout (simplified):
    # 0: wrist
    # 1-4: thumb (CMC, MCP, IP, TIP)
    # 5-8: index finger
    # 9-12: middle finger
    # 13-16: ring finger
    # 17-20: pinky

    for i in range(NUM_LANDMARKS):
        if i == 0:
            # Wrist
            x, y = wrist_x, wrist_y
        else:
            # Finger landmarks spread along the arm direction
            finger_group = (i - 1) // 4  # 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
            joint_index = (i - 1) % 4    # 0=base, 1=MCP, 2=IP/PIP, 3=TIP

            # Distance from wrist increases with joint index
            dist = 0.05 + joint_index * 0.04

            # Spread fingers perpendicular to arm direction
            spread = (finger_group - 2) * 0.02  # Center on middle finger
            perp_dx, perp_dy = -dy, dx  # Perpendicular to arm direction

            x = wrist_x + dx * dist + perp_dx * spread
            y = wrist_y + dy * dist + perp_dy * spread

        # Add noise
        x += random.gauss(0, noise_level)
        y += random.gauss(0, noise_level)
        z = random.gauss(0, 0.01)  # Depth noise

        landmarks.append((round(x, 6), round(y, 6), round(z, 6)))

    return landmarks


def main():
    output_path = os.path.join(os.path.dirname(__file__), "semaphore_landmarks.csv")

    # Column headers: letter, then landmark_0_x, landmark_0_y, landmark_0_z, ..., landmark_20_z
    headers = ["letter"]
    for i in range(NUM_LANDMARKS):
        headers.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

    rows = []
    for letter, (left_angle, right_angle) in sorted(SEMAPHORE_ANGLES.items()):
        for _ in range(SAMPLES_PER_LETTER):
            # We use the right hand angle as the primary signal
            # (in real semaphore both hands matter, but for a single-hand classifier
            # we use the right hand position)
            landmarks = generate_hand_landmarks(right_angle, noise_level=0.015)
            row = [letter]
            for lm in landmarks:
                row.extend(lm)
            rows.append(row)

    # Shuffle to mix letters
    random.shuffle(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated semaphore landmarks: {len(rows)} samples -> {output_path}")
    print(f"Letters: {sorted(SEMAPHORE_ANGLES.keys())}")
    print(f"Samples per letter: {SAMPLES_PER_LETTER}")
    print(f"Features per sample: {NUM_LANDMARKS * 3} (21 landmarks x 3 coords)")


if __name__ == "__main__":
    main()
