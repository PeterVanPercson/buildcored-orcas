"""
BUILDCORED ORCAS — Day 25: FirmwarePatcher
Load firmware binary. Detect patterns. LLM annotates.

Hardware concept: Firmware Analysis
Ghidra, binwalk, and hexdump do exactly this.
Every embedded security researcher and firmware
engineer reads binary blobs looking for magic numbers,
strings, memory maps, and hardcoded credentials.

Planted patterns in firmware_blob.bin:
  0x0000: ARM Cortex-M vector table
  0x0040: "ORCA" magic number + version
  0x0200: String table (including a hidden credential!)
  0x0300: Simulated register map
  0x03C0: 0xDEADBEEF fill pattern
  0x0FF8: CRC checksum + "END" marker

YOUR TASK:
1. Find the hidden credential in the string table (TODO #1)
2. Add a new pattern detector (TODO #2)

Run: python day25_starter.py [path/to/firmware.bin]
Default: looks for assets/firmware_blob.bin from Day 0 repo
"""

import sys
import os
import struct
import re
import subprocess
import time

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)

MODEL = "qwen2.5:3b"

# ============================================================
# FIND FIRMWARE BLOB
# ============================================================

def find_firmware():
    """Look for the firmware blob in common locations."""
    candidates = [
        "firmware_blob.bin",
        "assets/firmware_blob.bin",
        "../../assets/firmware_blob.bin",
        "../../../assets/firmware_blob.bin",
    ]

    if len(sys.argv) > 1:
        candidates.insert(0, sys.argv[1])

    for path in candidates:
        if os.path.exists(path):
            return path

    # Generate a fresh one if not found
    console.print("[yellow]firmware_blob.bin not found. Generating...[/yellow]")
    return generate_firmware_blob()


def generate_firmware_blob():
    """Regenerate the firmware blob in the current directory."""
    import random
    random.seed(42)
    blob = bytearray()

    # ARM vector table
    blob += struct.pack("<I", 0x20004000)
    blob += struct.pack("<I", 0x08000101)
    blob += struct.pack("<I", 0x08000201)
    blob += struct.pack("<I", 0x08000301)
    blob += b"\x00" * 52

    # Magic header at 0x40
    blob += b"ORCA"
    blob += struct.pack("<HHI", 1, 5, 4096)
    blob += struct.pack("<I", 0x20250325)
    blob += b"buildcored-fw\x00"
    blob += b"\x00" * (0x0080 - len(blob))

    # Code section
    for _ in range(96):
        blob += struct.pack("<H", random.randint(0x2000, 0xFFFF))
    blob += b"\x00" * (0x0200 - len(blob))

    # String table at 0x200
    strings = [
        b"ORCAS Firmware v1.5\x00",
        b"[INFO] System initialized\x00",
        b"[ERR] Sensor timeout\x00",
        b"[WARN] Low battery\x00",
        b"ADC_CHANNEL_0\x00",
        b"I2C_ADDR_0x68\x00",
        b"PWM_FREQ_1000\x00",
        b"UART_BAUD_115200\x00",
        b"admin:orcas2025\x00",
        b"GPIO_PIN_CONFIG\x00",
    ]
    for s in strings:
        blob += s
    blob += b"\x00" * (0x0300 - len(blob))

    # Register map
    for addr, val in [
        (0x40010000, 0x00000003),
        (0x40012000, 0x00000068),
        (0x40013000, 0x000003E8),
        (0x40020000, 0x0001C200),
    ]:
        blob += struct.pack("<II", addr, val)
    blob += b"\x00" * (0x03C0 - len(blob))

    # DEADBEEF fill
    blob += b"\xDE\xAD\xBE\xEF" * 12
    blob += b"\xFF" * (0x0FF8 - len(blob))

    # CRC + END
    checksum = sum(blob) & 0xFFFFFFFF
    blob += struct.pack("<I", checksum)
    blob += b"END\x00"

    path = "firmware_blob.bin"
    with open(path, "wb") as f:
        f.write(blob)
    console.print(f"[green]✓ Generated: {path} ({len(blob)} bytes)[/green]")
    return path


# ============================================================
# HEX DUMP
# ============================================================

def hex_dump(data, offset=0, length=None, bytes_per_row=16):
    """
    Render a hex dump as a rich Table.
    Format: OFFSET | HEX BYTES | ASCII
    """
    if length is None:
        length = len(data)

    t = Table(box=box.SIMPLE, show_header=True,
              header_style="bold cyan")
    t.add_column("Offset", style="dim", width=8)
    t.add_column("Hex", width=49)
    t.add_column("ASCII", width=18)

    for row_start in range(0, min(length, len(data) - offset), bytes_per_row):
        abs_offset = offset + row_start
        row_bytes = data[abs_offset: abs_offset + bytes_per_row]

        hex_str = " ".join(f"{b:02x}" for b in row_bytes)
        hex_str = hex_str.ljust(bytes_per_row * 3 - 1)

        ascii_str = "".join(
            chr(b) if 0x20 <= b < 0x7f else "."
            for b in row_bytes
        )

        t.add_row(f"0x{abs_offset:04X}", hex_str, ascii_str)

    return t


# ============================================================
# PATTERN DETECTION
# ============================================================

PATTERNS = {
    "ARM Vector Table": {
        "offset": 0x0000,
        "length": 64,
        "description": "ARM Cortex-M vector table. Stack pointer at 0x00, reset handler at 0x04.",
        "detect": lambda d: struct.unpack("<I", d[0:4])[0] > 0x20000000,
    },
    "Magic Number": {
        "offset": 0x0040,
        "length": 32,
        "description": "Firmware magic number 'ORCA' + version + build date.",
        "detect": lambda d: d[0x40:0x44] == b"ORCA",
    },
    "String Table": {
        "offset": 0x0200,
        "length": 256,
        "description": "Null-terminated ASCII strings. Firmware labels, error messages, and config keys.",
        "detect": lambda d: any(c for c in d[0x200:0x210] if 0x20 <= c < 0x7f),
    },
    "Register Map": {
        "offset": 0x0300,
        "length": 64,
        "description": "Memory-mapped peripheral registers. GPIO, I2C, TIM, UART base addresses.",
        "detect": lambda d: struct.unpack("<I", d[0x300:0x304])[0] > 0x40000000,
    },
    "Fill Pattern": {
        "offset": 0x03C0,
        "length": 32,
        "description": "0xDEADBEEF fill pattern. Classic embedded debug marker for uninitialized memory.",
        "detect": lambda d: d[0x3C0:0x3C4] == b"\xDE\xAD\xBE\xEF",
    },
    "Flash Erased": {
        "offset": 0x03F0,
        "length": 16,
        "description": "0xFF fill — erased flash. ARM flash erases to 0xFF.",
        "detect": lambda d: all(b == 0xFF for b in d[0x03F0:0x0400]),
    },
    "CRC Checksum": {
        "offset": 0x0FF8,
        "length": 8,
        "description": "CRC32 checksum + END marker. Firmware integrity verification.",
        "detect": lambda d: len(d) > 0x0FFC and d[0x0FFC:0x1000] == b"END\x00",
    },
}

# TODO #2: Add a new pattern detector
# Ideas:
#   "Null Strings":  find all null-terminated ASCII strings
#   "Jump Table":    find consecutive 32-bit pointers to code (thumb bit = odd address)
#   "Entropy":       measure byte entropy per 256-byte block (high entropy = encrypted/compressed)
#   "Repeated Bytes": find runs of identical bytes > 8 long


# TODO #1: Hidden credential finder
def find_credentials(data):
    """
    Scan the firmware for hardcoded credentials.
    Look for patterns like user:password or key=value.
    The starter already planted 'admin:orcas2025' in the string table.
    Can you find it using a regex?
    """
    # Search for ASCII strings that look like credentials
    text = ""
    for b in data:
        if 0x20 <= b < 0x7f:
            text += chr(b)
        else:
            text += " "

    # Pattern: word:word (user:password style)
    creds = re.findall(r"\b([a-zA-Z0-9_]{3,20}):([a-zA-Z0-9_!@#$%]{4,20})\b", text)
    return creds


# ============================================================
# LLM ANNOTATION
# ============================================================

def ask_llm_about_section(section_name, hex_preview, description):
    """Get LLM commentary on a firmware section."""
    prompt = (
        f"You are a firmware reverse engineer. Analyze this section of a binary firmware:\n\n"
        f"Section: {section_name}\n"
        f"Known info: {description}\n"
        f"Hex preview: {hex_preview}\n\n"
        f"In 2 sentences: what does this section do and why does it matter for security or functionality?"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except:
        return "[LLM unavailable]"


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_firmware(data):
    """Run full firmware analysis."""
    console.print(f"\n[bold cyan]Firmware size: {len(data)} bytes "
                  f"({len(data)/1024:.1f} KB)[/bold cyan]\n")

    # Detect patterns
    found = {}
    for name, pat in PATTERNS.items():
        try:
            if pat["detect"](data):
                found[name] = pat
        except Exception:
            pass

    console.print(f"[green]✓ Found {len(found)}/{len(PATTERNS)} known patterns[/green]\n")

    # Credential scan
    creds = find_credentials(data)
    if creds:
        console.print(Panel(
            "\n".join(f"  [red]⚠ Credential found:[/red] {u}:{p}" for u, p in creds),
            title="🔑 Hardcoded Credentials",
            border_style="red"
        ))
    else:
        console.print("[dim]No obvious credentials found[/dim]")

    console.print()

    # Annotate each section
    for name, pat in found.items():
        offset = pat["offset"]
        length = pat["length"]
        section_data = data[offset: offset + length]

        # Hex preview for LLM
        hex_preview = " ".join(f"{b:02x}" for b in section_data[:24])

        console.print(Panel(
            f"[dim]Offset: 0x{offset:04X} | Length: {length} bytes[/dim]\n"
            f"[white]{pat['description']}[/white]",
            title=f"[cyan]{name}[/cyan]",
            border_style="cyan"
        ))

        # Hex dump of section
        console.print(hex_dump(data, offset=offset, length=min(length, 48)))

        # LLM commentary
        console.print("[dim]🧠 Asking brain...[/dim]", end="")
        commentary = ask_llm_about_section(name, hex_preview, pat["description"])
        console.print(f"\r[bold green]🧠 Brain:[/bold green] {commentary}\n")

    # Full hex dump option
    console.print("\n[dim]Showing first 256 bytes of firmware:[/dim]")
    console.print(hex_dump(data, offset=0, length=256))


# ============================================================
# MAIN
# ============================================================

def main():
    console.print("\n[bold cyan]🔧 FirmwarePatcher — Day 25[/bold cyan]")
    console.print("[dim]Firmware analysis with LLM annotation[/dim]\n")

    # Load firmware
    path = find_firmware()
    console.print(f"[green]✓ Loading: {path}[/green]")

    with open(path, "rb") as f:
        data = f.read()

    console.print(f"[dim]Loaded {len(data)} bytes[/dim]\n")

    # Run analysis
    analyze_firmware(data)

    console.print("\n[bold]Analysis complete.[/bold]")
    console.print("[dim]Try: add your own pattern detector (TODO #2)[/dim]")
    console.print("\nSee you tomorrow for Day 26!")


if __name__ == "__main__":
    main()
