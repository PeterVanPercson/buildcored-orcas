"""
BUILDCORED ORCAS — Day 26: CacheSim
Two-level CPU cache simulator with LRU eviction.
Visualize hits, misses, and evictions in real time.

Hardware concept: Cache Hierarchy + Locality
L1: small, fast (~1ns). L2: larger, slower (~5ns).
RAM: huge, slow (~100ns). Cache misses stall the CPU.
Understanding this is essential for fast firmware.

v2.0 bridge: In v2.0, you'll profile real memory access
patterns on ARM Cortex-M0+ and compare predictions here.

YOUR TASK:
1. Generate a 0% L1 hit rate pattern (TODO #1)
2. Generate a 100% L1 hit rate pattern (TODO #2)
3. Explain why each works (TODO #3 — in your README)

Run: python day26_starter.py
"""

import time
import sys
import random
import collections

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)


# ============================================================
# CACHE CONFIGURATION
# ============================================================

# L1 Cache: small and fast (like a real Cortex-M L1)
L1_SIZE = 4       # Number of cache lines in L1
L1_LATENCY = 1    # Cycles (relative)

# L2 Cache: larger but slower
L2_SIZE = 8       # Number of cache lines in L2
L2_LATENCY = 5    # Cycles (relative)

# RAM: everything else
RAM_LATENCY = 20  # Cycles (relative)


# ============================================================
# CACHE IMPLEMENTATION
# ============================================================

class CacheLine:
    """One entry in a cache."""
    def __init__(self, address, timestamp):
        self.address = address
        self.timestamp = timestamp  # For LRU eviction


class Cache:
    """
    Direct-mapped or fully-associative cache with LRU eviction.
    We use fully-associative (any line can hold any address)
    for simplicity — real caches are set-associative.
    """

    def __init__(self, name, size, latency):
        self.name = name
        self.size = size
        self.latency = latency
        self.lines = {}        # address → CacheLine
        self.access_count = 0
        self.hit_count = 0
        self.eviction_count = 0
        self.clock = 0         # Logical timestamp for LRU

    def access(self, address):
        """
        Try to find address in cache.
        Returns (hit, evicted_address_or_None).
        LRU: evict the least recently used line when full.
        """
        self.clock += 1
        self.access_count += 1

        if address in self.lines:
            # Cache HIT — update timestamp (most recently used)
            self.lines[address].timestamp = self.clock
            self.hit_count += 1
            return True, None

        # Cache MISS — need to load from next level
        evicted = None
        if len(self.lines) >= self.size:
            # Cache is full — evict LRU line
            lru_addr = min(self.lines, key=lambda a: self.lines[a].timestamp)
            evicted = lru_addr
            del self.lines[lru_addr]
            self.eviction_count += 1

        # Load new line
        self.lines[address] = CacheLine(address, self.clock)
        return False, evicted

    def hit_rate(self):
        if self.access_count == 0:
            return 0.0
        return self.hit_count / self.access_count * 100

    def reset_stats(self):
        self.access_count = 0
        self.hit_count = 0
        self.eviction_count = 0

    def state_snapshot(self):
        """Return list of (address, recency) tuples, sorted MRU first."""
        return sorted(
            self.lines.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )


class TwoLevelCache:
    """L1 + L2 cache hierarchy with stats tracking."""

    def __init__(self):
        self.l1 = Cache("L1", L1_SIZE, L1_LATENCY)
        self.l2 = Cache("L2", L2_SIZE, L2_LATENCY)
        self.total_cycles = 0
        self.access_log = []  # (address, result, cycles)

    def access(self, address):
        """
        Access a memory address through the hierarchy.
        L1 hit  → 1 cycle
        L2 hit  → 5 cycles
        RAM hit → 20 cycles
        """
        # Try L1
        l1_hit, l1_evicted = self.l1.access(address)
        if l1_hit:
            self.total_cycles += L1_LATENCY
            self.access_log.append((address, "L1_HIT", L1_LATENCY))
            return "L1_HIT", L1_LATENCY

        # L1 miss — try L2
        l2_hit, l2_evicted = self.l2.access(address)
        if l2_hit:
            self.total_cycles += L2_LATENCY
            self.access_log.append((address, "L2_HIT", L2_LATENCY))
            return "L2_HIT", L2_LATENCY

        # L2 miss — go to RAM
        self.total_cycles += RAM_LATENCY
        self.access_log.append((address, "MISS", RAM_LATENCY))
        return "MISS", RAM_LATENCY

    def reset(self):
        self.l1 = Cache("L1", L1_SIZE, L1_LATENCY)
        self.l2 = Cache("L2", L2_SIZE, L2_LATENCY)
        self.total_cycles = 0
        self.access_log = []


# ============================================================
# VISUALIZATION
# ============================================================

RESULT_COLORS = {
    "L1_HIT": "green",
    "L2_HIT": "yellow",
    "MISS":   "red",
}

RESULT_LABELS = {
    "L1_HIT": "L1 ✓",
    "L2_HIT": "L2 ✓",
    "MISS":   "MISS",
}


def render_dashboard(cache, access_log, pattern_name):
    """Render the full cache simulator dashboard."""

    # ---- Access log table ----
    log_table = Table(title="Access Log (last 20)",
                      box=box.SIMPLE, show_header=True,
                      header_style="bold white")
    log_table.add_column("#", width=4, justify="right")
    log_table.add_column("Addr", width=6, justify="center")
    log_table.add_column("Result", width=8, justify="center")
    log_table.add_column("Cycles", width=7, justify="right")

    for i, (addr, result, cycles) in enumerate(access_log[-20:]):
        color = RESULT_COLORS[result]
        log_table.add_row(
            f"{len(access_log) - 20 + i + 1 if len(access_log) > 20 else i + 1}",
            f"0x{addr:03X}",
            f"[{color}]{RESULT_LABELS[result]}[/{color}]",
            f"[{color}]{cycles}[/{color}]"
        )

    # ---- Cache state table ----
    cache_table = Table(title="Cache State",
                        box=box.SIMPLE, show_header=True,
                        header_style="bold white")
    cache_table.add_column("Slot", width=5, justify="right")
    cache_table.add_column("L1 Addr", width=9, justify="center")
    cache_table.add_column("L2 Addr", width=9, justify="center")

    l1_state = cache.l1.state_snapshot()
    l2_state = cache.l2.state_snapshot()
    max_rows = max(L1_SIZE, L2_SIZE)

    for i in range(max_rows):
        l1_str = f"[green]0x{l1_state[i][0]:03X}[/green]" if i < len(l1_state) else "[dim]---[/dim]"
        l2_str = f"[yellow]0x{l2_state[i][0]:03X}[/yellow]" if i < len(l2_state) else "[dim]---[/dim]"
        cache_table.add_row(str(i), l1_str, l2_str)

    # ---- Stats panel ----
    l1_hr = cache.l1.hit_rate()
    l2_hr = cache.l2.hit_rate()
    total = len(access_log)
    l1_hits = cache.l1.hit_count
    l2_hits = cache.l2.hit_count
    misses = total - l1_hits - l2_hits
    miss_pct = misses / total * 100 if total > 0 else 0

    stats = (
        f"[bold]Pattern:[/bold] {pattern_name}\n"
        f"[bold]Accesses:[/bold] {total} | "
        f"[bold]Total cycles:[/bold] {cache.total_cycles}\n\n"
        f"[green]L1 Hit Rate: {l1_hr:.1f}%[/green]  "
        f"({l1_hits} hits, {L1_LATENCY} cycle each)\n"
        f"[yellow]L2 Hit Rate: {l2_hr:.1f}%[/yellow]  "
        f"({l2_hits} hits, {L2_LATENCY} cycles each)\n"
        f"[red]Miss Rate:   {miss_pct:.1f}%[/red]  "
        f"({misses} misses, {RAM_LATENCY} cycles each)\n\n"
        f"[dim]L1 evictions: {cache.l1.eviction_count} | "
        f"L2 evictions: {cache.l2.eviction_count}[/dim]\n"
        f"[dim]L1 size: {L1_SIZE} lines | L2 size: {L2_SIZE} lines[/dim]"
    )

    # ---- Legend ----
    legend = (
        "[green]■ L1 HIT[/green]  1 cycle\n"
        "[yellow]■ L2 HIT[/yellow]  5 cycles\n"
        "[red]■ MISS[/red]    20 cycles (RAM)"
    )

    return Panel(
        Columns([log_table, cache_table,
                 Panel(stats + "\n\n" + legend, border_style="dim")]),
        title="[bold cyan]CacheSim — Day 26[/bold cyan]",
        border_style="cyan"
    )


# ============================================================
# ACCESS PATTERNS
# ============================================================

def pattern_sequential(n=20, max_addr=16):
    """Sequential: 0,1,2,...,N — tests spatial locality."""
    return [i % max_addr for i in range(n)]


def pattern_repeated(n=20, addrs=[0, 1, 2, 3]):
    """Repeated: same few addresses over and over — temporal locality."""
    return [addrs[i % len(addrs)] for i in range(n)]


def pattern_strided(n=20, stride=5, max_addr=64):
    """Strided: 0,5,10,15... — cache-unfriendly if stride > cache size."""
    return [(i * stride) % max_addr for i in range(n)]


def pattern_random(n=20, max_addr=32, seed=42):
    """Random: worst case for cache."""
    random.seed(seed)
    return [random.randint(0, max_addr - 1) for _ in range(n)]


# ============================================================
# TODO #1: Design a 0% L1 hit rate pattern
# ============================================================
# For L1 to have 0% hits, every access must miss L1.
# L1 has L1_SIZE=4 lines. If you access more than 4 unique
# addresses in a repeating cycle, you'll evict before reuse.
#
# Hint: access 5+ unique addresses in rotation.
# Example: 0,1,2,3,4,0,1,2,3,4... with L1_SIZE=4
# Each time 0 is accessed, address 4 evicted it.
# Then 4 is needed again... but it was evicted by 0.
#

def pattern_zero_l1_hits(n=20):
    """TODO: Design this so L1 hit rate = 0%."""
    # Hint: cycle through L1_SIZE+1 unique addresses
    cycle = list(range(L1_SIZE + 1))  # [0,1,2,3,4]
    return [cycle[i % len(cycle)] for i in range(n)]


# ============================================================
# TODO #2: Design a 100% L1 hit rate pattern
# ============================================================
# For L1 to hit 100%, every access after the warmup must
# find its address in L1. After the first few cold misses
# (warming the cache), you need to stay within L1_SIZE
# unique addresses.
#
# Hint: only access L1_SIZE or fewer unique addresses.
#

def pattern_perfect_l1(n=20):
    """TODO: Design this so L1 hit rate approaches 100%."""
    # Access only 2 unique addresses — well within L1_SIZE=4
    # After 2 cold misses, every subsequent access is an L1 hit
    cycle = [0, 1]  # Only 2 unique addresses
    return [cycle[i % len(cycle)] for i in range(n)]


# ============================================================
# MAIN
# ============================================================

PATTERNS = {
    "1": ("Sequential (spatial locality)",  pattern_sequential()),
    "2": ("Repeated (temporal locality)",   pattern_repeated()),
    "3": ("Strided (cache-unfriendly)",     pattern_strided()),
    "4": ("Random (worst case)",            pattern_random()),
    "5": ("Zero L1 hits (TODO #1)",         pattern_zero_l1_hits()),
    "6": ("Perfect L1 hits (TODO #2)",      pattern_perfect_l1()),
}


def run_simulation(pattern_name, accesses, animate=True):
    """Run the cache simulation on a list of addresses."""
    cache = TwoLevelCache()

    if animate:
        console.print(f"\n[bold]Running: {pattern_name}[/bold]")
        console.print(f"[dim]Addresses: {accesses}[/dim]\n")

        with Live(render_dashboard(cache, [], pattern_name),
                  refresh_per_second=8, console=console) as live:
            for addr in accesses:
                result, cycles = cache.access(addr)
                live.update(render_dashboard(cache, cache.access_log, pattern_name))
                time.sleep(0.15)

        console.print()
    else:
        for addr in accesses:
            cache.access(addr)

    return cache


def main():
    console.print("\n[bold cyan]⚡ CacheSim — Day 26[/bold cyan]")
    console.print(f"[dim]L1: {L1_SIZE} lines @ {L1_LATENCY} cycle | "
                  f"L2: {L2_SIZE} lines @ {L2_LATENCY} cycles | "
                  f"RAM: {RAM_LATENCY} cycles[/dim]\n")

    while True:
        console.print("\n[bold]Choose an access pattern:[/bold]")
        for key, (name, _) in PATTERNS.items():
            console.print(f"  [{key}] {name}")
        console.print("  [q] Quit")
        console.print()

        try:
            choice = input("Choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == 'q':
            break

        if choice not in PATTERNS:
            console.print("[red]Invalid choice[/red]")
            continue

        name, accesses = PATTERNS[choice]
        cache = run_simulation(name, accesses, animate=True)

        # Summary
        console.print(Panel(
            f"[bold]Pattern:[/bold] {name}\n"
            f"[green]L1 hit rate: {cache.l1.hit_rate():.1f}%[/green]\n"
            f"[yellow]L2 hit rate: {cache.l2.hit_rate():.1f}%[/yellow]\n"
            f"[bold]Total cycles: {cache.total_cycles}[/bold]\n"
            f"[dim]L1 evictions: {cache.l1.eviction_count} | "
            f"L2 evictions: {cache.l2.eviction_count}[/dim]",
            title="Summary",
            border_style="cyan"
        ))

    console.print("\nCacheSim ended. See you tomorrow for Day 27!")


if __name__ == "__main__":
    main()
