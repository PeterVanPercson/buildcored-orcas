"""
BUILDCORED ORCAS — Day 23: ObjectFollower
Track a colored object. Compute PID error signal.
Output proportional controller commands.

Hardware concept: PID Control + Closed-Loop Feedback
The (x,y) offset IS the error signal for a servo PID.
P-term: proportional to distance from center.
In v2.0, this drives a real pan-tilt servo mount.

YOUR TASK:
1. Tune the HSV color range for your target (TODO #1)
2. Add I and D terms to the controller (TODO #2)
3. Run: python day23_starter.py

CONTROLS:
- Click on the object in the window to auto-sample its color
- 'r' → reset PID integrator
- 'q' → quit
"""

import cv2
import numpy as np
import sys
import time
import collections

# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

ret, frame = cap.read()
FRAME_H, FRAME_W = frame.shape[:2]
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2


# ============================================================
# TODO #1: HSV Color Range
# ============================================================
# HSV separates color (Hue) from brightness (Value) and
# saturation. This makes color thresholding much more robust
# than RGB — it works under different lighting conditions.
#
# OpenCV HSV ranges:
#   Hue:        0-179  (red=0/179, green=60, blue=120)
#   Saturation: 0-255  (0=gray, 255=vivid)
#   Value:      0-255  (0=black, 255=bright)
#
# Default below targets a bright green object.
# CLICK on your object in the window to auto-sample its color.
#
# Common targets:
#   Red:    H=(0-10 OR 170-179), S=(100-255), V=(100-255)
#   Green:  H=(40-80), S=(50-255), V=(50-255)
#   Blue:   H=(100-130), S=(50-255), V=(50-255)
#   Yellow: H=(20-35), S=(100-255), V=(100-255)
#   Orange: H=(10-20), S=(100-255), V=(100-255)

HSV_LOWER = np.array([40, 50, 50])    # <-- Adjust for your object
HSV_UPPER = np.array([80, 255, 255])   # <-- Adjust for your object

# Minimum contour area (pixels²) — filters out small noise blobs
MIN_CONTOUR_AREA = 500


# ============================================================
# PID CONTROLLER
# ============================================================

class PIDController:
    """
    Simple PID controller for one axis.

    Error = setpoint - current_value
    P term = Kp * error
    I term = Ki * integral(error, dt)
    D term = Kd * derivative(error, dt)
    Output = P + I + D
    """
    def __init__(self, Kp=0.5, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 0.001

        # P term
        p = self.Kp * error

        # TODO #2: Add I and D terms
        # I term accumulates error over time — fixes steady-state offset
        # D term damps oscillation — reacts to rate of change
        #
        # self.integral += error * dt
        # i = self.Ki * self.integral
        #
        # derivative = (error - self.prev_error) / dt
        # d = self.Kd * derivative
        #
        # For now I and D are zero:
        i = 0.0
        d = 0.0

        output = p + i + d

        self.prev_error = error
        self.prev_time = now

        return output, p, i, d

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# PID controllers for X and Y axes
pid_x = PIDController(Kp=0.5, Ki=0.0, Kd=0.0)
pid_y = PIDController(Kp=0.5, Ki=0.0, Kd=0.0)


# ============================================================
# COLOR SAMPLER (click to set target color)
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global HSV_LOWER, HSV_UPPER
    if event == cv2.EVENT_LBUTTONDOWN:
        # Sample color at clicked pixel
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = frame_hsv[y, x]
        print(f"\n📍 Sampled HSV at ({x},{y}): H={h}, S={s}, V={v}")

        # Create range around sampled color
        h_margin, s_margin, v_margin = 15, 60, 60
        HSV_LOWER = np.array([
            max(0, h - h_margin),
            max(0, s - s_margin),
            max(0, v - v_margin)
        ])
        HSV_UPPER = np.array([
            min(179, h + h_margin),
            min(255, s + s_margin),
            min(255, v + v_margin)
        ])
        print(f"   New range: lower={HSV_LOWER}, upper={HSV_UPPER}")

cv2.namedWindow("ObjectFollower - Day 23")
cv2.setMouseCallback("ObjectFollower - Day 23", mouse_callback)


# ============================================================
# TRACKING HISTORY (for trajectory visualization)
# ============================================================

trajectory = collections.deque(maxlen=50)
error_history_x = collections.deque(maxlen=100)
error_history_y = collections.deque(maxlen=100)


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def draw_crosshair(frame, x, y, size=20, color=(0, 255, 255), thickness=2):
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(frame, (x, y), size // 2, color, 1)


def draw_error_vector(frame, cx, cy, error_x, error_y):
    """Draw arrow from frame center to object center."""
    end_x = int(CENTER_X + error_x * 0.5)
    end_y = int(CENTER_Y + error_y * 0.5)
    cv2.arrowedLine(frame, (CENTER_X, CENTER_Y),
                    (cx, cy), (0, 100, 255), 2, tipLength=0.2)


def draw_pid_bars(frame, output_x, output_y, max_val=200):
    """Draw PID output as horizontal bars."""
    bar_y = FRAME_H - 60
    bar_h = 20

    # X axis bar
    bar_len = int(min(abs(output_x) / max_val * (FRAME_W // 3), FRAME_W // 3))
    color = (0, 180, 255) if output_x > 0 else (255, 100, 0)
    if output_x > 0:
        cv2.rectangle(frame, (CENTER_X, bar_y),
                      (CENTER_X + bar_len, bar_y + bar_h), color, -1)
    else:
        cv2.rectangle(frame, (CENTER_X - bar_len, bar_y),
                      (CENTER_X, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (0, bar_y), (FRAME_W, bar_y + bar_h), (80, 80, 80), 1)
    cv2.putText(frame, f"PID_X: {output_x:+.1f}",
                (10, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Y axis bar
    bar_y2 = FRAME_H - 35
    bar_len_y = int(min(abs(output_y) / max_val * (FRAME_H // 3), FRAME_H // 3))
    color_y = (0, 255, 100) if output_y > 0 else (255, 0, 100)
    if output_y > 0:
        cv2.rectangle(frame, (CENTER_X, bar_y2),
                      (CENTER_X + bar_len_y, bar_y2 + bar_h), color_y, -1)
    else:
        cv2.rectangle(frame, (CENTER_X - bar_len_y, bar_y2),
                      (CENTER_X, bar_y2 + bar_h), color_y, -1)
    cv2.rectangle(frame, (0, bar_y2), (FRAME_W, bar_y2 + bar_h), (80, 80, 80), 1)
    cv2.putText(frame, f"PID_Y: {output_y:+.1f}",
                (10, bar_y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


# ============================================================
# MAIN LOOP
# ============================================================

print("\n" + "=" * 55)
print("  🎯 ObjectFollower — Day 23")
print("=" * 55)
print()
print("  Click on your target object to sample its color.")
print("  Default: bright green objects")
print("  'r' = reset PID | 'q' = quit")
print()

frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Draw frame center crosshair
    draw_crosshair(display, CENTER_X, CENTER_Y,
                   size=30, color=(100, 100, 100), thickness=1)
    cv2.circle(display, (CENTER_X, CENTER_Y), 5, (200, 200, 200), -1)

    # Convert to HSV and threshold
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    object_found = False
    cx, cy = CENTER_X, CENTER_Y
    error_x, error_y = 0, 0
    output_x, output_y = 0.0, 0.0

    if contours:
        # Keep largest contour only
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_CONTOUR_AREA:
            object_found = True

            # Compute centroid
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # Bounding box
            bx, by, bw, bh = cv2.boundingRect(largest)
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh),
                          (0, 255, 0), 2)
            cv2.drawContours(display, [largest], -1, (0, 200, 0), 1)

            # Draw object centroid
            draw_crosshair(display, cx, cy, size=15,
                           color=(0, 255, 255), thickness=2)

            # Compute error (pixels from frame center)
            error_x = cx - CENTER_X
            error_y = cy - CENTER_Y

            # PID update
            output_x, px, ix, dx = pid_x.update(error_x)
            output_y, py, iy, dy = pid_y.update(error_y)

            # Draw error vector
            draw_error_vector(display, cx, cy, error_x, error_y)

            # Store trajectory
            trajectory.append((cx, cy))

            # Overlay text
            cv2.putText(display, f"Error X: {error_x:+d} px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display, f"Error Y: {error_y:+d} px",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display, f"Area: {area:.0f} px²",
                        (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1)

    else:
        pid_x.reset()
        pid_y.reset()

    # Draw trajectory
    pts = list(trajectory)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        color = (int(255 * alpha), int(200 * alpha), 0)
        cv2.line(display, pts[i-1], pts[i], color, 2)

    # Draw PID output bars
    draw_pid_bars(display, output_x, output_y)

    # Status
    status = "TRACKING" if object_found else "SEARCHING..."
    color = (0, 255, 0) if object_found else (0, 0, 255)
    cv2.putText(display, status, (FRAME_W - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # HSV range reminder
    cv2.putText(display,
                f"HSV: [{HSV_LOWER[0]}-{HSV_UPPER[0]}, "
                f"{HSV_LOWER[1]}-{HSV_UPPER[1]}, "
                f"{HSV_LOWER[2]}-{HSV_UPPER[2]}]",
                (10, FRAME_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(display, "Click=sample color | r=reset | q=quit",
                (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Show mask inset
    mask_small = cv2.resize(mask, (FRAME_W // 4, FRAME_H // 4))
    mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    display[10:10 + FRAME_H // 4, FRAME_W - FRAME_W // 4 - 10:FRAME_W - 10] = mask_color
    cv2.putText(display, "MASK",
                (FRAME_W - FRAME_W // 4 - 10, FRAME_H // 4 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    cv2.imshow("ObjectFollower - Day 23", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        pid_x.reset()
        pid_y.reset()
        trajectory.clear()
        print("PID reset.")

cap.release()
cv2.destroyAllWindows()
print("\nObjectFollower ended. See you tomorrow for Day 24!")
