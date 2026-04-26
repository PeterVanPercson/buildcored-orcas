“””
BUILDCORED ORCAS — Day 20: SensorLogger
Multi-channel DAQ — your laptop as a sensor array.

Hardware concept: Data Acquisition (DAQ)
NI-DAQ boards do exactly this: multiple sensors,
shared clock, ring buffer, configurable sample rate.
Your laptop IS the DAQ board today.

YOUR TASK:

1. Add a 4th sensor channel (TODO #1)
1. Tune anomaly detection threshold (TODO #2)
1. Run: python day20_starter.py
1. Press ‘s’ to export CSV, ‘q’ to quit
   “””

import time
import threading
import collections
import csv
import sys
import os
from datetime import datetime

import numpy as np
import psutil
import cv2
import pyaudio

try:
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich import box
console = Console()
except ImportError:
print(“pip install rich”); sys.exit(1)

# ============================================================

# CONFIGURATION

# ============================================================

SAMPLE_RATE_HZ = 10       # Samples per second (every 100ms)
RING_BUFFER_SIZE = 200    # Keep last 200 samples per channel
DISPLAY_HISTORY = 20      # How many samples to show in sparkline

# ============================================================

# RING BUFFER

# ============================================================

class SensorChannel:
“”“One channel of the DAQ — value, timestamp, history.”””
def **init**(self, name, unit, color=“white”):
self.name = name
self.unit = unit
self.color = color
self.values = collections.deque(maxlen=RING_BUFFER_SIZE)
self.timestamps = collections.deque(maxlen=RING_BUFFER_SIZE)
self.lock = threading.Lock()

```
def push(self, value):
    with self.lock:
        self.values.append(float(value))
        self.timestamps.append(time.time())

def latest(self):
    with self.lock:
        return self.values[-1] if self.values else 0.0

def history(self, n=DISPLAY_HISTORY):
    with self.lock:
        vals = list(self.values)[-n:]
    return vals

def mean(self):
    with self.lock:
        vals = list(self.values)
    return float(np.mean(vals)) if vals else 0.0

def std(self):
    with self.lock:
        vals = list(self.values)
    return float(np.std(vals)) if len(vals) > 1 else 0.0
```

# ============================================================

# SENSOR CHANNELS

# ============================================================

channels = {
“mic_rms”:    SensorChannel(“Mic RMS”,       “amplitude”, “cyan”),
“cam_motion”: SensorChannel(“Cam Motion”,    “px/frame”,  “green”),
“cpu_pct”:    SensorChannel(“CPU %”,         “%”,         “yellow”),
“mem_pct”:    SensorChannel(“Memory %”,      “%”,         “magenta”),
}

# TODO #1: Add a 4th sensor channel

# Ideas:

# - “keystroke_rate”: count keypresses per second (use pynput)

# - “disk_io”: psutil.disk_io_counters().read_bytes

# - “net_bytes”: psutil.net_io_counters().bytes_sent + bytes_recv

# - “battery_pct”: psutil.sensors_battery().percent (None on desktops)

# - “screen_brightness”: varies by OS (skip if complex)

# 

# Add it to the channels dict and create a reader thread for it.

# ============================================================

# SENSOR READERS

# ============================================================

running = True

# — Mic RMS —

def mic_reader():
try:
pa = pyaudio.PyAudio()
dev = None
for i in range(pa.get_device_count()):
if pa.get_device_info_by_index(i)[‘maxInputChannels’] > 0:
dev = i; break
if dev is None: return
stream = pa.open(format=pyaudio.paFloat32, channels=1,
rate=16000, input=True,
input_device_index=dev,
frames_per_buffer=1600)
while running:
try:
data = stream.read(1600, exception_on_overflow=False)
samples = np.frombuffer(data, dtype=np.float32)
rms = float(np.sqrt(np.mean(samples**2)))
channels[“mic_rms”].push(rms)
except: pass
time.sleep(1.0 / SAMPLE_RATE_HZ)
stream.stop_stream(); stream.close()
pa.terminate()
except Exception as e:
while running:
channels[“mic_rms”].push(0.0)
time.sleep(1.0 / SAMPLE_RATE_HZ)

# — Camera Motion (optical flow magnitude) —

def cam_reader():
cap = cv2.VideoCapture(0)
if not cap.isOpened():
cap = cv2.VideoCapture(1)
if not cap.isOpened():
while running:
channels[“cam_motion”].push(0.0)
time.sleep(1.0 / SAMPLE_RATE_HZ)
return

```
prev_gray = None
while running:
    ret, frame = cap.read()
    if not ret:
        channels["cam_motion"].push(0.0)
        time.sleep(1.0 / SAMPLE_RATE_HZ)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 120))  # Small = fast

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = float(np.mean(np.sqrt(
            flow[..., 0]**2 + flow[..., 1]**2
        )))
        channels["cam_motion"].push(magnitude)
    else:
        channels["cam_motion"].push(0.0)

    prev_gray = gray
    time.sleep(1.0 / SAMPLE_RATE_HZ)

cap.release()
```

# — CPU + Memory (psutil) —

def system_reader():
while running:
channels[“cpu_pct”].push(psutil.cpu_percent(interval=None))
channels[“mem_pct”].push(psutil.virtual_memory().percent)
time.sleep(1.0 / SAMPLE_RATE_HZ)

# ============================================================

# ANOMALY DETECTION

# ============================================================

# TODO #2: Tune these thresholds

# An anomaly fires when the latest value is more than

# ANOMALY_SIGMA standard deviations from the channel’s mean.

# Higher = fewer alerts. Lower = more sensitive.

ANOMALY_SIGMA = 2.5

anomaly_log = collections.deque(maxlen=5)

def check_anomalies():
“”“Flag channels where latest value > mean + ANOMALY_SIGMA * std.”””
alerts = []
for name, ch in channels.items():
if len(ch.values) < 20:
continue
val = ch.latest()
mean = ch.mean()
std = ch.std()
if std > 1e-6 and abs(val - mean) > ANOMALY_SIGMA * std:
direction = “↑” if val > mean else “↓”
alerts.append(f”{ch.name} {direction} {val:.3f}”)
return alerts

# ============================================================

# SPARKLINE

# ============================================================

def sparkline(values, width=20):
“”“ASCII sparkline from a list of values.”””
chars = “ ▁▂▃▄▅▆▇█”
if not values or max(values) == min(values):
return “─” * width
lo, hi = min(values), max(values)
result = “”
for v in values[-width:]:
idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
result += chars[idx]
return result.ljust(width)

# ============================================================

# RICH DASHBOARD

# ============================================================

def build_dashboard():
“”“Build the rich table dashboard.”””
t = Table(
title=“SensorLogger — DAQ Dashboard”,
box=box.ROUNDED,
show_header=True,
header_style=“bold cyan”,
)
t.add_column(“Channel”, style=“bold”, width=14)
t.add_column(“Latest”, justify=“right”, width=10)
t.add_column(“Mean”, justify=“right”, width=10)
t.add_column(“Std”, justify=“right”, width=8)
t.add_column(“Trend”, width=22)
t.add_column(“Status”, width=8)

```
for name, ch in channels.items():
    val = ch.latest()
    mean = ch.mean()
    std = ch.std()
    hist = ch.history()
    spark = sparkline(hist)

    # Anomaly check
    is_anomaly = (std > 1e-6 and
                  len(ch.values) >= 20 and
                  abs(val - mean) > ANOMALY_SIGMA * std)
    status = "[red]⚠ SPIKE[/red]" if is_anomaly else "[green]OK[/green]"

    t.add_row(
        f"[{ch.color}]{ch.name}[/{ch.color}]",
        f"[{ch.color}]{val:.4f}[/{ch.color}] {ch.unit}",
        f"{mean:.4f}",
        f"{std:.4f}",
        f"[{ch.color}]{spark}[/{ch.color}]",
        status,
    )

# Anomaly panel
alerts = check_anomalies()
if alerts:
    anomaly_log.extend(alerts)

stats_text = (
    f"[dim]Samples: {min(len(ch.values) for ch in channels.values())} | "
    f"Buffer: {RING_BUFFER_SIZE} | "
    f"Rate: {SAMPLE_RATE_HZ} Hz | "
    f"Press 's' to export CSV, 'q' to quit[/dim]"
)

if anomaly_log:
    alert_text = "[red]⚠ Recent anomalies: " + " | ".join(list(anomaly_log)[-3:]) + "[/red]"
else:
    alert_text = "[green]No anomalies detected[/green]"

from rich.text import Text
from rich import print as rprint

return Panel(
    t,
    subtitle=f"{stats_text}\n{alert_text}",
    border_style="cyan",
)
```

# ============================================================

# CSV EXPORT

# ============================================================

def export_csv():
“”“Export all channel data to a timestamped CSV.”””
fname = f”sensorlog_{datetime.now().strftime(’%H%M%S’)}.csv”
with open(fname, “w”, newline=””) as f:
writer = csv.writer(f)

```
    # Header
    header = ["timestamp"]
    for name in channels:
        header.append(name)
    writer.writerow(header)

    # Align channels by index (they may have different lengths)
    min_len = min(len(ch.timestamps) for ch in channels.values())
    for i in range(min_len):
        row = [list(list(channels.values())[0].timestamps)[i]]
        for ch in channels.values():
            row.append(list(ch.values)[i])
        writer.writerow(row)

console.print(f"\n[green]✓ Exported {fname} ({min_len} rows)[/green]")
return fname
```

# ============================================================

# MAIN

# ============================================================

def main():
global running

```
console.print("\n[bold cyan]📊 SensorLogger — Day 20[/bold cyan]")
console.print("[dim]Starting sensor threads...[/dim]\n")

# Start reader threads
threads = [
    threading.Thread(target=mic_reader, daemon=True),
    threading.Thread(target=cam_reader, daemon=True),
    threading.Thread(target=system_reader, daemon=True),
]
for t in threads:
    t.start()

# Wait for initial data
time.sleep(1.5)

console.print("[green]✓ All sensors active[/green]")
console.print("[dim]Press 's' + Enter to export CSV. Ctrl+C to quit.[/dim]\n")

# Input listener for commands
export_flag = threading.Event()

def input_listener():
    while running:
        try:
            cmd = input().strip().lower()
            if cmd == 's':
                export_flag.set()
            elif cmd == 'q':
                global running
                running = False
        except: break

input_thread = threading.Thread(target=input_listener, daemon=True)
input_thread.start()

try:
    with Live(build_dashboard(), refresh_per_second=4,
              console=console) as live:
        while running:
            live.update(build_dashboard())

            if export_flag.is_set():
                export_csv()
                export_flag.clear()

            time.sleep(0.25)

except KeyboardInterrupt:
    pass
finally:
    running = False

console.print("\n[bold]SensorLogger ended.[/bold]")
console.print("See you tomorrow for Day 21!")
```

if **name** == “**main**”:
main()
