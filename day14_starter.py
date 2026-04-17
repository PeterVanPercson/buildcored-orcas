"""
BUILDCORED ORCAS — Day 14: RegisterBot
Tiny CPU simulator with 8 registers, ALU, and instruction set.
Local LLM narrates each step.

Hardware concept: Fetch-Decode-Execute cycle, registers, ALU.
This IS computer architecture.

TASKS:
1. Implement the instruction decoder (TODO #1)
2. Tune the LLM narrator prompt (TODO #2)

Run: python day14_starter.py
PREREQS: ollama serve + ollama pull qwen2.5:3b
"""
import subprocess, sys, time
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)

MODEL = "qwen2.5:3b"

# ====== CPU STATE ======
class CPU:
    def __init__(self):
        self.registers = [0] * 8       # R0-R7
        self.pc = 0                     # Program counter
        self.flags = {"ZERO": False, "NEG": False}  # Compare flags
        self.halted = False
        self.program = []

    def reset(self):
        self.__init__()

# ====== ALU (pre-built) ======
def alu(op, a, b):
    """Arithmetic Logic Unit — does the math."""
    if op == "ADD": return a + b
    if op == "SUB": return a - b
    if op == "MUL": return a * b
    if op == "AND": return a & b
    if op == "OR":  return a | b
    raise ValueError(f"Unknown ALU op: {op}")

# ====== TODO #1: Instruction Decoder ======
# Given an instruction like ["MOV", "R0", "5"] or ["ADD", "R1", "R2"],
# execute it on the CPU.
#
# Instructions to implement:
#   MOV Rx, <value>    → registers[x] = value
#   MOV Rx, Ry         → registers[x] = registers[y]
#   ADD Rx, Ry         → registers[x] = registers[x] + registers[y]
#   SUB Rx, Ry         → registers[x] = registers[x] - registers[y]
#   MUL Rx, Ry         → registers[x] = registers[x] * registers[y]
#   CMP Rx, Ry         → set flags (ZERO if equal, NEG if Rx < Ry)
#   JMP <addr>         → pc = addr
#   JZ  <addr>         → if ZERO flag: pc = addr
#   JNZ <addr>         → if NOT ZERO flag: pc = addr
#   HALT               → cpu.halted = True
#
# Arguments can be "R0"-"R7" (register) or a number (immediate).
# Helper: reg_index("R3") → 3,  parse_arg() returns int value.

def reg_index(s):
    """Convert 'R0' to 0, 'R1' to 1, etc."""
    return int(s[1:])

def is_register(s):
    return s.startswith("R") and len(s) >= 2 and s[1:].isdigit()

def get_value(cpu, arg):
    """If arg is a register name, return its value. Otherwise parse as int."""
    if is_register(arg):
        return cpu.registers[reg_index(arg)]
    return int(arg)

def execute(cpu, instruction):
    """Execute ONE instruction. Returns a human-readable description."""
    op = instruction[0].upper()
    args = instruction[1:]

    if op == "MOV":
        dst = reg_index(args[0])
        val = get_value(cpu, args[1])
        cpu.registers[dst] = val
        cpu.pc += 1
        return f"R{dst} ← {val}"

    elif op in ("ADD", "SUB", "MUL"):
        dst = reg_index(args[0])
        b = get_value(cpu, args[1])
        a = cpu.registers[dst]
        cpu.registers[dst] = alu(op, a, b)
        cpu.pc += 1
        return f"R{dst} = {a} {op} {b} = {cpu.registers[dst]}"

    elif op == "CMP":
        a = get_value(cpu, args[0])
        b = get_value(cpu, args[1])
        cpu.flags["ZERO"] = (a == b)
        cpu.flags["NEG"] = (a < b)
        cpu.pc += 1
        return f"CMP {a} vs {b} → ZERO={cpu.flags['ZERO']}, NEG={cpu.flags['NEG']}"

    elif op == "JMP":
        cpu.pc = int(args[0])
        return f"JMP → line {cpu.pc}"

    elif op == "JZ":
        if cpu.flags["ZERO"]:
            cpu.pc = int(args[0])
            return f"JZ taken → line {cpu.pc}"
        else:
            cpu.pc += 1
            return "JZ not taken"

    elif op == "JNZ":
        if not cpu.flags["ZERO"]:
            cpu.pc = int(args[0])
            return f"JNZ taken → line {cpu.pc}"
        else:
            cpu.pc += 1
            return "JNZ not taken"

    elif op == "HALT":
        cpu.halted = True
        return "HALT — CPU stopped"

    else:
        cpu.pc += 1
        return f"Unknown op: {op}"

# ====== LLM NARRATOR ======

# TODO #2: Tune this prompt so narration is concise and educational
NARRATOR_PROMPT = """You are a CPU architecture teacher narrating one instruction step.

Instruction: {instr}
Effect: {effect}
Registers after: {regs}

In ONE short sentence, explain what just happened in hardware terms
(e.g. "The ALU added R1 and R2, storing the result in R0's register.").
No preamble. One sentence."""

def narrate(instr, effect, regs):
    try:
        r = subprocess.run(
            ["ollama", "run", MODEL,
             NARRATOR_PROMPT.format(instr=" ".join(instr), effect=effect, regs=regs)],
            capture_output=True, text=True, timeout=20
        )
        return r.stdout.strip()
    except:
        return ""

# ====== DISPLAY ======
def show_state(cpu, step, instr, effect, narration):
    t = Table(title=f"Step {step}: {' '.join(instr)}", show_header=True)
    for i in range(8):
        t.add_column(f"R{i}", justify="center")
    t.add_row(*[str(r) for r in cpu.registers])
    console.print(t)
    console.print(f"  [cyan]Effect:[/cyan] {effect}")
    console.print(f"  [dim]Flags: ZERO={cpu.flags['ZERO']}, NEG={cpu.flags['NEG']}, PC={cpu.pc}[/dim]")
    if narration:
        console.print(Panel(narration, title="🧠 Narrator", border_style="green"))
    console.print()

# ====== DEMO PROGRAM: Factorial of 5 ======
# Computes 5! = 120 using a loop
# R0 = counter (starts at 5), R1 = result (starts at 1)
PROGRAM = [
    ["MOV", "R0", "5"],      # 0: counter = 5
    ["MOV", "R1", "1"],      # 1: result = 1
    ["MOV", "R2", "0"],      # 2: zero constant for CMP
    ["MOV", "R3", "1"],      # 3: one constant for decrement
    ["CMP", "R0", "R2"],     # 4: is counter == 0?
    ["JZ",  "9"],             # 5: if yes, jump to HALT
    ["MUL", "R1", "R0"],     # 6: result *= counter
    ["SUB", "R0", "R3"],     # 7: counter -= 1
    ["JMP", "4"],             # 8: loop back
    ["HALT"],                 # 9: done
]

# ====== MAIN ======
console.print("\n[bold cyan]🖥️  RegisterBot — Tiny CPU Simulator[/bold cyan]")
console.print("[dim]Computing 5! using a loop[/dim]\n")

cpu = CPU()
cpu.program = PROGRAM
step = 0

while not cpu.halted and cpu.pc < len(cpu.program) and step < 50:
    instr = cpu.program[cpu.pc]
    step += 1

    # Capture state
    effect = execute(cpu, instr)
    regs_str = ",".join(f"R{i}={v}" for i, v in enumerate(cpu.registers) if v != 0)

    # Narrate
    narration = narrate(instr, effect, regs_str)

    show_state(cpu, step, instr, effect, narration)
    time.sleep(0.3)

console.print(f"[bold green]✓ Program finished in {step} steps[/bold green]")
console.print(f"[bold]Final result: R1 = {cpu.registers[1]}[/bold]  (5! should equal 120)")
console.print("\nSee you tomorrow for Day 15!")
