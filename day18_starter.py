"""
BUILDCORED ORCAS — Day 18: DepthMapper
Live monocular depth estimation using MiDaS.
Depth heatmap, point cloud CSV export, histogram.

Hardware concept: Depth Sensing + Point Clouds
ToF sensors, structured light cameras, and LiDAR all
produce depth maps. You're building the software version
using monocular AI depth estimation.

YOUR TASK:
1. Add center-region distance estimation (TODO #1)
2. Tune the colormap for better depth visualization (TODO #2)
3. Export a point cloud CSV on keypress (already built — understand it)

Run: python day18_starter.py
PREREQS: pip install torch torchvision (or pip install onnxruntime)
         First run downloads MiDaS-small model (~80 MB)

CONTROLS:
- s → save point cloud CSV
- h → show depth histogram
- q → quit
"""

import cv2
import numpy as np
import sys
import os
import time

# ============================================================
# MODEL LOADING — tries torch first, then ONNX fallback
# ============================================================

MODEL_TYPE = None
depth_model = None
depth_transform = None

def load_midas_torch():
    global depth_model, depth_transform, MODEL_TYPE
    try:
        import torch
        print("Loading MiDaS-small via torch hub (~80 MB first run)...")
        depth_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small",
            trust_repo=True
        )
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            trust_repo=True
        )
        depth_transform = transforms.small_transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model.to(device)
        depth_model.eval()
        MODEL_TYPE = "torch"
        print(f"✓ MiDaS-small loaded (device: {device})")
        return True
    except Exception as e:
        print(f"torch/MiDaS failed: {e}")
        return False


def load_onnx_fallback():
    """
    Minimal fallback: use a simple gradient-based pseudo-depth
    if neither torch nor ONNX is available.
    This is NOT real depth estimation but produces a plausible
    visualization for demonstration.
    """
    global MODEL_TYPE
    MODEL_TYPE = "pseudo"
    print("⚠️  Using pseudo-depth fallback (gradient-based)")
    print("   Install torch for real depth: pip install torch torchvision")
    return True


print("\n" + "=" * 55)
print("  📡 DepthMapper — Day 18")
print("=" * 55)

if not load_midas_torch():
    load_onnx_fallback()


# ============================================================
# DEPTH INFERENCE
# ============================================================

def estimate_depth(frame_bgr):
    """
    Run depth estimation on a BGR frame.
    Returns a normalized depth map (0=near, 1=far or inverse).
    """
    if MODEL_TYPE == "torch":
        import torch
        device = next(depth_model.parameters()).device

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform (resizes and normalizes)
        input_batch = depth_transform(rgb).to(device)

        with torch.no_grad():
            prediction = depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

    else:
        # Pseudo-depth: use focus measure (Laplacian variance) per region
        # Blurry regions = far, sharp regions = near
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute local sharpness as proxy for depth
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        # Smooth to get region-level depth estimate
        depth = cv2.GaussianBlur(np.abs(laplacian), (51, 51), 0)

    # Normalize to 0-1
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth


# ============================================================
# TODO #2: Colormap selection
# ============================================================
# OpenCV colormaps turn a grayscale depth map into a color image.
# Different colormaps emphasize different depth features.
#
# Try changing this value to one of:
#   cv2.COLORMAP_PLASMA   — purple to yellow (good contrast)
#   cv2.COLORMAP_JET      — classic blue-to-red (the default)
#   cv2.COLORMAP_MAGMA    — black to white via pink (nice for dark scenes)
#   cv2.COLORMAP_INFERNO  — high contrast for nearby objects
#   cv2.COLORMAP_TURBO    — improved JET (more perceptually uniform)
#   cv2.COLORMAP_HOT      — black/red/yellow/white
#
COLORMAP = cv2.COLORMAP_PLASMA  # <-- Change this


def colorize_depth(depth_normalized):
    """Convert depth array to colorized heatmap."""
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, COLORMAP)


# ============================================================
# POINT CLOUD EXPORT
# ============================================================

def export_point_cloud(depth_map, frame_shape, filename="point_cloud.csv"):
    """
    Export depth map as a point cloud CSV.
    Each row: x (pixel), y (pixel), depth (normalized 0-1)

    In real LiDAR, you'd also have real-world XYZ in meters.
    Here we use pixel coordinates and normalized depth.
    """
    h, w = frame_shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    d_flat = depth_map.flatten()

    # Subsample (every 4th pixel) — full res = huge file
    step = 4
    xs_s = xs_flat[::step]
    ys_s = ys_flat[::step]
    ds_s = d_flat[::step]

    import csv
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "depth"])
        for x, y, d in zip(xs_s, ys_s, ds_s):
            writer.writerow([x, y, f"{d:.4f}"])

    total_points = len(xs_s)
    print(f"✓ Saved {filename} ({total_points:,} points)")
    return filename


# ============================================================
# DEPTH HISTOGRAM
# ============================================================

def show_histogram(depth_map):
    """Plot depth distribution histogram."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.hist(depth_map.flatten(), bins=100, color='#0f7173', edgecolor='none')
    plt.xlabel("Depth (normalized 0=near, 1=far)")
    plt.ylabel("Pixel count")
    plt.title("Depth Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# TODO #1: Center-region distance estimation
# ============================================================
# Compute the average depth value in the center 20% of the frame.
# Display it as a "range finder" reading.
#
# This mimics how a camera autofocus sensor or ultrasonic rangefinder
# reports distance from the center of its field of view.
#
# Returns a float 0-1 (0=very close, 1=very far relative to scene).
#

def estimate_center_depth(depth_map):
    """
    Get average depth in center 20% of frame.
    Returns float 0-1.
    """
    h, w = depth_map.shape
    cy, cx = h // 2, w // 2
    region_h, region_w = h // 5, w // 5

    region = depth_map[
        cy - region_h//2 : cy + region_h//2,
        cx - region_w//2 : cx + region_w//2
    ]
    return float(np.mean(region)) if region.size > 0 else 0.5


# ============================================================
# MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found."); sys.exit(1)

print("\nDepthMapper running!")
print("Controls: 's'=save CSV | 'h'=histogram | 'q'=quit")
print("Move your hand closer/further — watch the heatmap change.\n")

last_depth = None
frame_count = 0
fps_start = time.time()
fps = 0.0
PROCESS_EVERY = 3  # Run depth estimation every N frames (CPU is slow)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run depth estimation every PROCESS_EVERY frames
    if frame_count % PROCESS_EVERY == 0:
        depth = estimate_depth(frame)
        last_depth = depth

        # FPS estimate
        elapsed = time.time() - fps_start
        fps = PROCESS_EVERY / elapsed if elapsed > 0 else 0
        fps_start = time.time()

    if last_depth is None:
        continue

    # Colorize depth map
    heatmap = colorize_depth(last_depth)

    # Blend with original frame for context
    blended = cv2.addWeighted(frame, 0.3, heatmap, 0.7, 0)

    # Center region overlay
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rw, rh = w // 10, h // 10
    cv2.rectangle(blended, (cx-rw, cy-rh), (cx+rw, cy+rh), (255, 255, 255), 2)

    # Center depth value
    center_d = estimate_center_depth(last_depth)
    if center_d < 0.33:
        range_str = "NEAR"
        range_color = (0, 255, 0)
    elif center_d < 0.66:
        range_str = "MID"
        range_color = (0, 255, 255)
    else:
        range_str = "FAR"
        range_color = (0, 0, 255)

    cv2.putText(blended, f"Center: {range_str} ({center_d:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, range_color, 2)
    cv2.putText(blended, f"FPS: {fps:.1f} | Model: {MODEL_TYPE}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(blended, "s=save CSV  h=histogram  q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("DepthMapper - Day 18", blended)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and last_depth is not None:
        fname = export_point_cloud(last_depth, frame.shape)
        cv2.putText(blended, f"Saved: {fname}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif key == ord('h') and last_depth is not None:
        show_histogram(last_depth)

cap.release()
cv2.destroyAllWindows()
print("\nDepthMapper ended. See you tomorrow for Day 19!")
