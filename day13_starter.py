"""
BUILDCORED ORCAS — Day 13: DailyDebrief
Collect your day's activity and get an AI summary.

Hardware concept: Flight Data Recorder
Collect all streams → compress → report.

TASKS:
1. Tune the debrief prompt (TODO #1)
2. Add a 4th data source (TODO #2)

Run: python day13_starter.py
"""
import subprocess, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)

MODEL = "qwen2.5:3b"

def check_ollama():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if "qwen2.5" not in r.stdout.lower():
            console.print("[red]Run: ollama pull qwen2.5:3b[/red]"); sys.exit(1)
    except: console.print("[red]ollama not found[/red]"); sys.exit(1)

check_ollama()

# ====== DATA SOURCES ======

def get_git_commits(hours=24):
    """Last N hours of git commits from current repo."""
    try:
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        r = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=format:%h %s"],
            capture_output=True, text=True, timeout=5
        )
        commits = r.stdout.strip().split("\n") if r.stdout.strip() else []
        return commits[:20]
    except: return []

def get_recent_files(hours=24):
    """Files modified in the last N hours in home dir."""
    home = Path.home()
    cutoff = time.time() - (hours * 3600)
    recent = []
    for p in home.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime > cutoff:
                # Skip system/cache files
                if any(x in str(p) for x in [".cache", "node_modules", ".git/", "__pycache__", "Library"]):
                    continue
                recent.append(str(p.relative_to(home)))
                if len(recent) >= 30: break
        except: pass
    return recent

def get_shell_history(lines=30):
    """Last N lines of shell history."""
    for hist_file in [".zsh_history", ".bash_history"]:
        path = Path.home() / hist_file
        if path.exists():
            try:
                with open(path, "r", errors="ignore") as f:
                    all_lines = f.readlines()
                return [l.strip() for l in all_lines[-lines:] if l.strip()]
            except: pass
    return []

# TODO #2: Add a 4th source — browser bookmarks, VS Code workspace,
# Spotify plays, calendar events, etc.

# ====== LLM SUMMARY ======

# TODO #1: Tune this prompt
DEBRIEF_PROMPT = """Analyze this developer's day. Output EXACTLY 5 sections:

BUILT: [what they built, 1 line]
BROKE: [what went wrong, 1 line]
LEARNED: [what they likely learned, 1 line]
PATTERN: [biggest theme, 1 line]
NEXT: [suggested next step, 1 line]

Data:
{data}

5 sections only. No preamble:"""

def get_debrief(data_text):
    prompt = DEBRIEF_PROMPT.format(data=data_text[:3000])
    try:
        r = subprocess.run(["ollama", "run", MODEL, prompt],
                         capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except: return "[LLM error]"

# ====== MAIN ======

console.print("\n[bold cyan]📊 DailyDebrief[/bold cyan]\n")
console.print("[dim]Collecting data from the last 24 hours...[/dim]\n")

commits = get_git_commits()
files = get_recent_files()
history = get_shell_history()

console.print(f"  Git commits:    {len(commits)}")
console.print(f"  Recent files:   {len(files)}")
console.print(f"  Shell commands: {len(history)}")
console.print()

# Build data string for LLM
data = []
if commits:
    data.append("GIT COMMITS:\n" + "\n".join(commits[:10]))
if files:
    data.append("FILES MODIFIED:\n" + "\n".join(files[:15]))
if history:
    data.append("SHELL HISTORY:\n" + "\n".join(history[-15:]))

if not data:
    console.print("[yellow]No data found. Make some git commits first![/yellow]")
    sys.exit(0)

combined = "\n\n".join(data)

console.print("[dim]Asking the brain...[/dim]")
start = time.time()
debrief = get_debrief(combined)
elapsed = time.time() - start

console.print(Panel(debrief, title=f"Today's Debrief ({elapsed:.1f}s)",
                    border_style="cyan"))
console.print("\nSee you tomorrow for Day 14!")
