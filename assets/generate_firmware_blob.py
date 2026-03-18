#!/usr/bin/env python3
"""
Generate a simulated firmware binary blob for Day 25 (FirmwarePatcher).

This creates a realistic-looking binary with planted patterns that students
must find using hex analysis and LLM-assisted inspection:
- ARM Cortex-M vector table header
- Magic numbers at aligned offsets
- Null-terminated ASCII strings (including a hidden "password")
- Repeated byte patterns (like fill patterns)
- A simulated register map section
- CRC-like checksum at the end

Output: assets/firmware_blob.bin (~4 KB)
"""

import struct
import os
import random

random.seed(42)  # Reproducible output

blob = bytearray()

# --- Section 1: ARM Cortex-M Vector Table (0x0000 - 0x003F) ---
# Stack pointer initial value
blob += struct.pack("<I", 0x20004000)  # SP = top of 16KB SRAM
# Reset handler
blob += struct.pack("<I", 0x08000101)  # Reset vector (thumb bit set)
# NMI handler
blob += struct.pack("<I", 0x08000201)
# HardFault handler
blob += struct.pack("<I", 0x08000301)
# MemManage, BusFault, UsageFault
blob += struct.pack("<I", 0x08000401)
blob += struct.pack("<I", 0x08000501)
blob += struct.pack("<I", 0x08000601)
# Reserved (4 entries)
blob += b"\x00" * 16
# SVCall, Debug, Reserved, PendSV, SysTick
blob += struct.pack("<I", 0x08000701)
blob += struct.pack("<I", 0x00000000)
blob += struct.pack("<I", 0x00000000)
blob += struct.pack("<I", 0x08000801)
blob += struct.pack("<I", 0x08000901)

# --- Section 2: Magic number + firmware header (0x0040 - 0x007F) ---
blob += b"ORCA"  # Magic number
blob += struct.pack("<H", 1)   # Major version
blob += struct.pack("<H", 5)   # Minor version
blob += struct.pack("<I", 4096)  # Firmware size
blob += struct.pack("<I", 0x20250325)  # Build date (hex-encoded)
blob += b"buildcored-fw\x00"  # Firmware name (null-terminated)
blob += b"\x00" * (0x0080 - len(blob))  # Pad to 0x80

# --- Section 3: Simulated code section with random-looking data (0x0080 - 0x01FF) ---
# Mix of realistic ARM thumb instructions and random data
for _ in range(96):
    # Generate plausible-looking 16-bit thumb instructions
    blob += struct.pack("<H", random.randint(0x2000, 0xFFFF))

# Pad code section to 0x0200
blob += b"\x00" * (0x0200 - len(blob))

# --- Section 4: String table (0x0200 - 0x02FF) ---
strings = [
    b"ORCAS Firmware v1.5\x00",
    b"[INFO] System initialized\x00",
    b"[ERR] Sensor timeout\x00",
    b"[WARN] Low battery\x00",
    b"ADC_CHANNEL_0\x00",
    b"I2C_ADDR_0x68\x00",
    b"PWM_FREQ_1000\x00",
    b"UART_BAUD_115200\x00",
    b"admin:orcas2025\x00",  # <-- Hidden password for students to find
    b"GPIO_PIN_CONFIG\x00",
]

for s in strings:
    blob += s

# Pad to 0x0300
blob += b"\x00" * (0x0300 - len(blob))

# --- Section 5: Simulated register map / config block (0x0300 - 0x03BF) ---
# This looks like memory-mapped peripheral registers
register_block = [
    (0x4001_0000, 0x0000_0003),  # GPIO_MODER
    (0x4001_0004, 0x0000_0000),  # GPIO_OTYPER
    (0x4001_0008, 0x0000_000C),  # GPIO_OSPEEDR
    (0x4001_000C, 0x0000_0000),  # GPIO_PUPDR
    (0x4001_2000, 0x0000_0068),  # I2C_CR1 (address 0x68)
    (0x4001_2004, 0x0000_2710),  # I2C_CR2 (10kHz)
    (0x4001_3000, 0x0000_03E8),  # TIM_ARR (PWM period = 1000)
    (0x4001_3004, 0x0000_01F4),  # TIM_CCR1 (50% duty = 500)
    (0x4002_0000, 0x0001_C200),  # UART_BRR (115200 baud)
    (0x4002_0004, 0x0000_000D),  # UART_CR1 (TX+RX enable)
]

for addr, val in register_block:
    blob += struct.pack("<II", addr, val)

# Pad to 0x03C0
blob += b"\x00" * (0x03C0 - len(blob))

# --- Section 6: Fill pattern section (0x03C0 - 0x03EF) ---
# Repeating pattern that students should recognize
blob += b"\xDE\xAD\xBE\xEF" * 12  # Classic deadbeef fill pattern

# --- Section 7: Padding with 0xFF (like erased flash) (0x03F0 - 0x0FF7) ---
blob += b"\xFF" * (0x0FF8 - len(blob))

# --- Section 8: CRC-like checksum at the end (0x0FF8 - 0x0FFF) ---
# Simple checksum: sum of all bytes modulo 2^32
checksum = sum(blob) & 0xFFFFFFFF
blob += struct.pack("<I", checksum)
blob += b"END\x00"

# Write
output_path = os.path.join(os.path.dirname(__file__), "firmware_blob.bin")
with open(output_path, "wb") as f:
    f.write(blob)

print(f"Generated firmware blob: {len(blob)} bytes -> {output_path}")
print(f"Checksum: 0x{checksum:08X}")
print()
print("Planted patterns for students to find:")
print("  - ARM vector table at 0x0000")
print('  - Magic number "ORCA" at 0x0040')
print("  - Firmware version 1.5 at 0x0044")
print("  - Build date 0x20250325 at 0x004C")
print("  - String table starting at 0x0200")
print('  - Hidden credential "admin:orcas2025" in string table')
print("  - Register map at 0x0300 (GPIO, I2C, TIM, UART)")
print("  - DEADBEEF fill pattern at 0x03C0")
print("  - Checksum at 0x0FF8")
print('  - "END" marker at 0x0FFC')
