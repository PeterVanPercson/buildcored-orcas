“””
BUILDCORED ORCAS — Day 19: I2CPlayground
Complete I2C protocol simulation with animated waveforms.

Hardware concept: I2C Bus Protocol
I2C uses just 2 wires (SDA + SCL) to connect hundreds of
sensors to a microcontroller. After today you can read any
sensor datasheet and understand the timing diagrams.

KEY I2C FACTS:

- SDA: Serial Data line (bidirectional)
- SCL: Serial Clock line (driven by master)
- MSB first: most significant bit sent first
- ACK bit: 9th bit after each byte. LOW=ACK, HIGH=NACK
- START: SDA goes LOW while SCL is HIGH
- STOP:  SDA goes HIGH while SCL is HIGH

YOUR TASK:

1. Add a NACK simulation (TODO #1)
1. Implement clock stretching (TODO #2)
1. Run a multi-byte transaction (TODO #3)

Run: python day19_starter.py
“””

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
import time

# ============================================================

# I2C PARAMETERS

# ============================================================

SCL_FREQ = 100_000    # 100 kHz standard mode
SAMPLES_PER_BIT = 20  # How many plot points per bit period

# ============================================================

# I2C ENCODING

# ============================================================

def encode_i2c_transaction(device_addr, register_addr, data_bytes, read=False):
“””
Encode a complete I2C transaction as a list of (SDA, SCL) bit pairs.

```
Structure of a write transaction:
  START | ADDR(7) | R/W(1) | ACK | REG(8) | ACK | DATA(8) | ACK | STOP

I2C is MSB first — bit 7 goes on the wire before bit 0.

Returns: list of (label, sda_bits, scl_bits, annotation)
"""
segments = []

# --- START condition ---
# SDA goes LOW while SCL is HIGH
segments.append(("START", [1, 1, 0, 0], [1, 1, 1, 0], "START"))

# --- Address byte + R/W bit ---
# 7-bit address, MSB first, then R/W bit (0=write, 1=read)
addr_bits = [(device_addr >> (6 - i)) & 1 for i in range(7)]
rw_bit = 1 if read else 0
addr_byte = addr_bits + [rw_bit]
sda_seq, scl_seq = bits_to_waveform(addr_byte)
segments.append(("ADDR+RW", sda_seq, scl_seq,
                 f"ADDR=0x{device_addr:02X} {'R' if read else 'W'}"))

# --- ACK from slave after address ---
ack_sda, ack_scl = generate_ack(ack=True)
segments.append(("ACK", ack_sda, ack_scl, "ACK"))

# --- Register address byte ---
reg_bits = [(register_addr >> (7 - i)) & 1 for i in range(8)]
sda_seq, scl_seq = bits_to_waveform(reg_bits)
segments.append(("REG", sda_seq, scl_seq, f"REG=0x{register_addr:02X}"))

# --- ACK from slave ---
ack_sda, ack_scl = generate_ack(ack=True)
segments.append(("ACK", ack_sda, ack_scl, "ACK"))

# --- Data bytes ---
for i, byte in enumerate(data_bytes):
    data_bits = [(byte >> (7 - j)) & 1 for j in range(8)]
    sda_seq, scl_seq = bits_to_waveform(data_bits)
    segments.append((f"DATA[{i}]", sda_seq, scl_seq, f"DATA=0x{byte:02X}"))

    # ACK after each data byte (NACK after last byte in read)
    is_last = (i == len(data_bytes) - 1)
    ack = not (read and is_last)  # NACK on last byte in read
    ack_sda, ack_scl = generate_ack(ack=ack)
    label = "ACK" if ack else "NACK"
    segments.append((label, ack_sda, ack_scl, label))

# --- STOP condition ---
# SDA goes HIGH while SCL is HIGH
segments.append(("STOP", [0, 0, 1, 1], [0, 1, 1, 1], "STOP"))

return segments
```

def bits_to_waveform(bits):
“””
Convert a list of bit values to SDA and SCL waveform arrays.
SCL: clock pulse (low-high-low) for each bit
SDA: held stable during HIGH phase of SCL (I2C requirement)
“””
sda = []
scl = []
for bit in bits:
# Each bit: SCL low → SDA set → SCL high → SCL low
sda += [bit] * SAMPLES_PER_BIT
scl += [0] * (SAMPLES_PER_BIT // 4) +   
[1] * (SAMPLES_PER_BIT // 2) +   
[0] * (SAMPLES_PER_BIT // 4)
return sda, scl

def generate_ack(ack=True):
“””
Generate ACK (SDA LOW during SCL HIGH) or
NACK (SDA HIGH during SCL HIGH).
ACK signals the slave received the byte successfully.
NACK signals error or end of read.
“””
sda_val = 0 if ack else 1
sda = [sda_val] * SAMPLES_PER_BIT
scl = [0] * (SAMPLES_PER_BIT // 4) +   
[1] * (SAMPLES_PER_BIT // 2) +   
[0] * (SAMPLES_PER_BIT // 4)
return sda, scl

# ============================================================

# I2C DECODING

# ============================================================

def decode_i2c_segments(segments):
“””
Decode encoded I2C segments back to human-readable form.
Verifies MSB-first bit ordering and ACK/NACK states.
“””
decoded = []
for label, sda_bits, scl_bits, annotation in segments:
if label in (“START”, “STOP”):
decoded.append(f”[{label}]”)
elif label in (“ACK”, “NACK”):
decoded.append(f”  → {label}”)
else:
# Sample SDA during SCL HIGH phase for each bit
bits = []
i = 0
while i < len(sda_bits):
# Find a HIGH phase in SCL
if i < len(scl_bits) and scl_bits[i] == 1:
bits.append(sda_bits[i])
# Skip rest of this HIGH phase
while i < len(scl_bits) and scl_bits[i] == 1:
i += 1
else:
i += 1

```
        # Reconstruct byte value (MSB first)
        if len(bits) >= 8:
            byte_val = 0
            for bit in bits[:8]:
                byte_val = (byte_val << 1) | bit
            decoded.append(f"  {label}: 0x{byte_val:02X} = {byte_val:08b}b = {byte_val}d")
        else:
            decoded.append(f"  {label}: {bits}")

return decoded
```

# ============================================================

# TODO #1: NACK simulation

# ============================================================

# A slave sends NACK (SDA HIGH during ACK slot) when:

# - It doesn’t recognize the address

# - It’s busy (clock stretching pending)

# - A read transaction ends (master sends NACK on last byte)

# 

# Add a function simulate_nack() that:

# 1. Runs a normal write transaction

# 2. Forces a NACK instead of ACK after the address byte

# 3. Prints the decoded error

# 4. Shows what a “device not found” I2C error looks like

# 

# ============================================================

# TODO #2: Clock stretching

# ============================================================

# Slow slaves can hold SCL LOW to pause the master and buy

# processing time. The master must wait until SCL goes HIGH

# before continuing.

# 

# Simulate this by inserting extra LOW samples in the SCL

# waveform during one of the ACK slots. Label it “STRETCH”.

# 

# This is why I2C drivers have timeout logic — a stuck-low

# SCL bus means a slave crashed mid-transaction.

# 

# ============================================================

# TODO #3: Multi-byte transaction

# ============================================================

# Real sensors send multiple data bytes in one transaction.

# An MPU6050 sends 14 bytes: 3-axis accel, temp, 3-axis gyro.

# 

# Try encoding this simulated MPU6050 reading:

# Device addr: 0x68 (MPU6050 default)

# Register:    0x3B (ACCEL_XOUT_H, first data register)

# Data:        [0x03, 0xE8, 0xFF, 0x01, …] (14 bytes)

# 

# ============================================================

# WAVEFORM BUILDER

# ============================================================

def build_full_waveform(segments):
“”“Concatenate all segments into full SDA and SCL arrays.”””
all_sda = []
all_scl = []
boundaries = [0]  # Segment start indices
labels = []

```
for label, sda, scl, annotation in segments:
    all_sda.extend(sda)
    all_scl.extend(scl)
    boundaries.append(len(all_sda))
    labels.append((boundaries[-2], annotation))

return np.array(all_sda), np.array(all_scl), labels
```

# ============================================================

# ANIMATED VISUALIZATION

# ============================================================

def animate_transaction(segments, title=“I2C Transaction”):
“”“Animate the SDA and SCL waveforms scrolling left.”””
sda_full, scl_full, labels = build_full_waveform(segments)
N = len(sda_full)
WINDOW = 200  # Samples visible at once

```
fig, (ax_scl, ax_sda) = plt.subplots(2, 1, figsize=(12, 5),
                                      sharex=True)
fig.suptitle(title, fontsize=13, fontweight='bold')

# SCL (clock)
ax_scl.set_ylim(-0.3, 1.5)
ax_scl.set_ylabel("SCL", fontweight='bold')
ax_scl.set_yticks([0, 1])
ax_scl.set_yticklabels(["LOW", "HIGH"])
ax_scl.grid(True, alpha=0.2)
ax_scl.set_facecolor("#0d0d14")
scl_line, = ax_scl.step([], [], color='#f59e0b', linewidth=2, where='post')

# SDA (data)
ax_sda.set_ylim(-0.3, 1.5)
ax_sda.set_ylabel("SDA", fontweight='bold')
ax_sda.set_xlabel("Sample index")
ax_sda.set_yticks([0, 1])
ax_sda.set_yticklabels(["LOW", "HIGH"])
ax_sda.grid(True, alpha=0.2)
ax_sda.set_facecolor("#0d0d14")
sda_line, = ax_sda.step([], [], color='#22c55e', linewidth=2, where='post')

# Annotation text
ann_text = ax_scl.text(0.02, 1.2, "", transform=ax_scl.transAxes,
                       fontsize=10, color='white', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))

fig.patch.set_facecolor('#0a0a0f')
for ax in (ax_scl, ax_sda):
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

def update(frame):
    start = min(frame * 4, N - WINDOW)
    end = start + WINDOW
    x = np.arange(start, end)

    # Pad if needed
    scl_slice = scl_full[start:end]
    sda_slice = sda_full[start:end]
    if len(scl_slice) < WINDOW:
        scl_slice = np.pad(scl_slice, (0, WINDOW - len(scl_slice)))
        sda_slice = np.pad(sda_slice, (0, WINDOW - len(sda_slice)))

    scl_line.set_data(x, scl_slice)
    sda_line.set_data(x, sda_slice)
    ax_scl.set_xlim(start, end)
    ax_sda.set_xlim(start, end)

    # Find current annotation
    current_label = ""
    for pos, lbl in labels:
        if pos <= start + WINDOW // 2:
            current_label = lbl
    ann_text.set_text(current_label)

    return scl_line, sda_line, ann_text

frames = max(1, (N - WINDOW) // 4 + 20)
ani = animation.FuncAnimation(
    fig, update, frames=frames,
    interval=60, blit=False, repeat=True
)

plt.tight_layout()
plt.show()
return ani
```

# ============================================================

# TERMINAL VISUALIZATION (fallback if matplotlib is slow)

# ============================================================

def print_transaction(segments):
“”“Print I2C transaction as text with ASCII timing diagram.”””
print(”\n” + “=” * 60)
print(”  I2C Transaction (text mode)”)
print(”=” * 60)

```
for label, sda_bits, scl_bits, annotation in segments:
    # Show a simplified waveform slice (12 chars)
    sda_display = "".join(str(b) for b in sda_bits[:12])
    scl_display = "".join(str(b) for b in scl_bits[:12])
    print(f"  {label:<10} | SCL: {scl_display}... | SDA: {sda_display}... | {annotation}")

print()
print("Decoded:")
decoded = decode_i2c_segments(segments)
for line in decoded:
    print(f"  {line}")
print()
```

# ============================================================

# MAIN

# ============================================================

def main():
print(”\n” + “=” * 60)
print(”  🔌 I2CPlayground — Day 19”)
print(”=” * 60)
print()

```
# Example: Write 0x42 to register 0x1A of device at address 0x48
# (Simulating a temperature sensor, e.g., LM75 at 0x48)
DEVICE_ADDR = 0x48    # LM75 temperature sensor default address
REGISTER    = 0x1A    # Example config register
DATA        = [0x42]  # Example data byte

print(f"  Encoding write transaction:")
print(f"  Device: 0x{DEVICE_ADDR:02X} | Register: 0x{REGISTER:02X} | Data: {[hex(b) for b in DATA]}")
print()

segments = encode_i2c_transaction(DEVICE_ADDR, REGISTER, DATA, read=False)

# Terminal print
print_transaction(segments)

# Decode and verify
print("Verification (decoder output):")
decoded = decode_i2c_segments(segments)
for line in decoded:
    print(f"  {line}")
print()

print("Key I2C facts:")
print("  - ACK = SDA pulled LOW by slave during 9th bit")
print("  - NACK = SDA stays HIGH (slave didn't respond)")
print("  - START: SDA↓ while SCL is HIGH")
print("  - STOP:  SDA↑ while SCL is HIGH")
print("  - MSB first: bit7 goes on wire before bit0")
print()

# Animated visualization
print("Launching animated waveform...")
print("(Close the plot window to continue)\n")
animate_transaction(
    segments,
    title=f"I2C Write: Device=0x{DEVICE_ADDR:02X} Reg=0x{REGISTER:02X} Data=0x{DATA[0]:02X}"
)

print("\nSee you tomorrow for Day 20!")
```

if **name** == “**main**”:
main()
