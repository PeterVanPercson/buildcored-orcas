"""
BUILDCORED ORCAS — Day 17: PWMSimulator
Interactive PWM simulator: slider controls duty cycle,
square wave animates live, virtual LED dims accordingly.

Hardware concept: Pulse Width Modulation (PWM)
Every microcontroller fakes analog output using PWM.
A digital pin rapidly switches on/off. The RATIO of
on-time to off-time (duty cycle) sets the average voltage.
Dimmable LEDs, servos, motors — all PWM.

v2.0 bridge: On a Pico, pwm.duty_u16(value) does this
in hardware. Today you simulate it. In v2.0 you DRIVE it.

YOUR TASK:
1. Add a second PWM channel at a different frequency (TODO #1)
2. Understand why duty cycle = average voltage (TODO #2)

Run: python day17_starter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.patches import Circle

# ============================================================
# PWM PARAMETERS
# ============================================================

VCC = 3.3             # Supply voltage (Pico uses 3.3V)
PWM_FREQ = 1000       # PWM frequency in Hz (1 kHz — typical for LEDs)
DISPLAY_CYCLES = 5    # How many PWM cycles to show on screen
SAMPLE_POINTS = 1000  # Resolution of the waveform plot

# Current duty cycle (set by slider, 0-100%)
current_duty = 50.0


# ============================================================
# PWM WAVEFORM GENERATION
# ============================================================

def generate_pwm_wave(duty_percent, freq=PWM_FREQ, cycles=DISPLAY_CYCLES, points=SAMPLE_POINTS):
    """
    Generate a PWM square wave.
    Returns (time_array, voltage_array).

    Duty cycle = (time HIGH) / (total period)
    If duty = 50%, the wave is HIGH half the time.
    If duty = 25%, the wave is HIGH for 1/4 of each period.
    """
    period = 1.0 / freq
    total_time = cycles * period
    t = np.linspace(0, total_time, points)

    # For each time point, figure out where we are in the current period
    position_in_period = (t % period) / period  # 0.0 to 1.0

    # HIGH when position is less than duty fraction
    duty_fraction = duty_percent / 100.0
    voltage = np.where(position_in_period < duty_fraction, VCC, 0.0)

    return t, voltage


def compute_average_voltage(duty_percent):
    """
    The average voltage of a PWM signal.

    # ============================================================
    # TODO #2: Understand why this math works
    # ============================================================
    # Average = (V_high * time_high + V_low * time_low) / total_time
    # For PWM: V_low = 0, time_high / total_time = duty_fraction
    # So: average = VCC * duty_fraction
    #
    # This is why a dimmable LED at 50% duty looks half-bright:
    # the AVERAGE voltage (what the human eye perceives) is VCC/2.
    # The LED is actually fully ON or fully OFF — never in between.
    """
    return VCC * (duty_percent / 100.0)


# ============================================================
# TODO #1: Add a second PWM channel
# ============================================================
# Real microcontrollers have MULTIPLE independent PWM channels.
# The Pico has 16 of them!
#
# Try adding a second channel at a DIFFERENT frequency (e.g. 500 Hz)
# with its own duty cycle slider. This teaches:
#   - PWM channels are independent
#   - Different frequencies produce different waveforms visually
#   - Phase relationships between channels matter
#
# For simplicity the starter shows one channel. Extend it.


# ============================================================
# PLOT SETUP
# ============================================================

fig = plt.figure(figsize=(12, 7))
fig.suptitle("PWMSimulator — Day 17", fontsize=14, fontweight='bold')

# Main waveform plot (top, wide)
ax_wave = plt.axes([0.08, 0.45, 0.65, 0.45])
ax_wave.set_xlim(0, DISPLAY_CYCLES / PWM_FREQ * 1000)  # x axis in ms
ax_wave.set_ylim(-0.3, VCC + 0.3)
ax_wave.set_xlabel("Time (ms)")
ax_wave.set_ylabel("Voltage (V)")
ax_wave.set_title("PWM Waveform")
ax_wave.grid(True, alpha=0.3)

# Waveform line
wave_line, = ax_wave.plot([], [], color='#43a047', linewidth=2)

# Average voltage line (horizontal dashed)
avg_line = ax_wave.axhline(y=VCC/2, color='#ff6f00',
                           linestyle='--', linewidth=1.5, label='Average V')
ax_wave.legend(loc='upper right')

# Virtual LED visualization (right side)
ax_led = plt.axes([0.78, 0.45, 0.18, 0.45])
ax_led.set_xlim(-1.5, 1.5)
ax_led.set_ylim(-1.5, 1.5)
ax_led.set_aspect('equal')
ax_led.axis('off')
ax_led.set_title("Virtual LED", fontweight='bold')

# LED body (glows based on duty)
led_circle = Circle((0, 0), 1.0, color='#ffeb3b', alpha=0.5)
ax_led.add_patch(led_circle)

# LED glow ring
led_glow = Circle((0, 0), 1.3, color='#ffeb3b', alpha=0.2)
ax_led.add_patch(led_glow)

# LED base/socket
led_base = plt.Rectangle((-0.6, -1.4), 1.2, 0.3, color='#424242')
ax_led.add_patch(led_base)

# Info text (below LED)
info_text = ax_led.text(0, -1.8, "",
                        ha='center', fontsize=10, fontweight='bold',
                        color='#333')

# Slider (bottom)
slider_ax = plt.axes([0.15, 0.22, 0.7, 0.04])
duty_slider = Slider(
    ax=slider_ax,
    label="Duty Cycle (%)",
    valmin=0, valmax=100,
    valinit=current_duty,
    valstep=1,
    color='#0f7173',
)

# Stats panel (bottom)
stats_ax = plt.axes([0.08, 0.03, 0.84, 0.12])
stats_ax.axis('off')
stats_text = stats_ax.text(
    0.5, 0.5, "",
    ha='center', va='center', fontsize=11,
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#f0f0f0', pad=1),
    transform=stats_ax.transAxes
)


# ============================================================
# UPDATE LOGIC
# ============================================================

def update_display(duty_percent):
    """Refresh everything based on new duty cycle."""
    global current_duty
    current_duty = duty_percent

    # Update waveform
    t, v = generate_pwm_wave(duty_percent)
    wave_line.set_data(t * 1000, v)  # Convert to ms

    # Update average line
    avg_v = compute_average_voltage(duty_percent)
    avg_line.set_ydata([avg_v, avg_v])

    # Update LED brightness
    # Alpha based on duty cycle (0 = off, 1 = full bright)
    brightness = duty_percent / 100.0
    led_circle.set_alpha(0.15 + brightness * 0.85)
    led_glow.set_alpha(brightness * 0.4)

    # LED color shifts from dim-red-orange to bright-yellow as duty rises
    # (like a real incandescent behavior, approximately)
    if brightness < 0.3:
        led_circle.set_color('#ef6c00')  # Dim orange
    elif brightness < 0.7:
        led_circle.set_color('#ffa000')  # Amber
    else:
        led_circle.set_color('#ffeb3b')  # Bright yellow

    # Info text under LED
    if brightness == 0:
        state = "OFF"
    elif brightness < 0.3:
        state = "DIM"
    elif brightness < 0.7:
        state = "MEDIUM"
    else:
        state = "BRIGHT"
    info_text.set_text(state)

    # Update stats
    period_us = 1e6 / PWM_FREQ
    time_high_us = period_us * brightness
    time_low_us = period_us - time_high_us
    stats = (
        f"PWM Frequency: {PWM_FREQ} Hz | Period: {period_us:.0f} µs  |  "
        f"HIGH time: {time_high_us:.0f} µs  |  LOW time: {time_low_us:.0f} µs\n"
        f"Duty Cycle: {duty_percent:.0f}%  |  "
        f"Average voltage: {avg_v:.2f} V  |  "
        f"Supply (VCC): {VCC} V"
    )
    stats_text.set_text(stats)

    fig.canvas.draw_idle()


# Hook slider to update function
duty_slider.on_changed(update_display)

# Initial render
update_display(current_duty)

# ============================================================
# RUN
# ============================================================

print("\n" + "=" * 55)
print("  💡 PWMSimulator — Day 17")
print("=" * 55)
print()
print("  Drag the slider to change duty cycle.")
print("  Watch the square wave, average voltage, and LED change.")
print()
print("  Hardware fact: your CPU does this at ~10,000,000 Hz")
print("  using nothing but a digital output pin.")
print()
print("  Close the plot window to quit.")
print()

plt.show()

print("\nPWMSimulator ended. See you tomorrow for Day 18!")
