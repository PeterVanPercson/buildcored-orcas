# BUILDCORED ORCAS

**30 Days. 30 Projects. Zero Hardware Required.**

A daily build challenge that teaches hardware engineering thinking through software projects you can run on any laptop.

| | |
|---|---|
| **Projects** | 30 |
| **Weeks** | 4 |
| **Time per day** | ~1 hour |
| **Cloud dependencies** | 0 |

Works on Mac, Windows, and Linux. All AI runs locally — no API keys, no cloud, no cost.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/buildcored-orcas.git
cd buildcored-orcas
python verify_setup.py
```

Follow the [Environment Setup Guide](SETUP_GUIDE.md) to install all dependencies before Day 1.

## What Is This?

BUILDCORED ORCAS is a 30-day daily build challenge for people with Python fundamentals who want to develop hardware engineering intuition — without owning any hardware.

Every day, you build a complete, working project in about an hour. Each project targets a real hardware concept — PWM, I2C, ADC, cache architecture, interrupt handlers, PID control — implemented entirely in software on your laptop.

**The core mental model:** sensor → process → output. Your webcam is a sensor. Your microphone is an ADC. Your screen is an actuator.

## The 4 Weeks

| Week | Theme | Focus |
|------|-------|-------|
| 1 | Body as Input | Webcam + mic as sensors. Gesture, gaze, breath → digital actions |
| 2 | Local AI Core | On-device LLMs via ollama. Edge inference, memory budgets, latency |
| 3 | Signals & Systems | FFT, filters, PWM, I2C, DAQ — hardware fundamentals in software |
| 4 | Full Systems | Sensor → model → actuator pipelines. System integration |

## What You Need

- A laptop (Mac, Windows, or Linux) with a webcam and microphone
- Python 3.10+
- About 1 hour per day for 30 consecutive days
- Willingness to ship imperfect code daily

## Submission Format

Every project must include a completed README using the [template](templates/README_TEMPLATE.md). This is part of the "shipped" definition.

## What Comes Next: ORCAS v2.0

After 30 days of software-first thinking, v2.0 puts real chips in your hands. Raspberry Pi Pico W, real sensors, real actuators. The PID controller from Day 23 drives a real servo. The I2C protocol from Day 19 reads a real accelerometer. The PWM from Day 17 dims a real LED.

v1.5 is the foundation. v2.0 is where it becomes physical.

## Community

BUILDCORED ORCAS runs as a cohort challenge on Discord and Telegram. You'll have a squad of 5 people, a squad lead for daily support, and mentors for technical guidance.

## Repository Structure

```
buildcored-orcas/
├── verify_setup.py          # Run this first — checks your environment
├── SETUP_GUIDE.md           # Full setup instructions
├── assets/
│   ├── semaphore_landmarks.csv   # Baseline dataset for Day 27 (SignalFlags)
│   ├── firmware_blob.bin         # Simulated firmware for Day 25 (FirmwarePatcher)
│   └── datasheets/               # Component datasheets for Day 24 (HardwareTA)
├── templates/
│   └── README_TEMPLATE.md        # Required README format for every project
└── days/                         # Day-specific resources (populated during challenge)
```
