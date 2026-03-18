# Component Datasheets

These datasheets are used in **Day 24 (HardwareTA)** — a RAG agent that answers hardware engineering questions by indexing and searching these PDFs.

## Required Datasheets

Download these open-source datasheets and place them in this folder:

| Component | Manufacturer | Download From |
|-----------|-------------|---------------|
| ESP32-WROOM-32 | Espressif | https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32_datasheet_en.pdf |
| STM32F103C8 ("Blue Pill") | STMicroelectronics | https://www.st.com/resource/en/datasheet/stm32f103c8.pdf |
| DHT22 / AM2302 | Aosong | Search "DHT22 datasheet PDF" — available from multiple sources |
| MPU-6050 (Accelerometer/Gyro) | InvenSense/TDK | https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf |
| SG90 Micro Servo | TowerPro | Search "SG90 servo datasheet PDF" — widely available |

## Why These Specific Components?

These are the exact components used in **ORCAS v2.0** (the hardware track):

- **ESP32** — Common WiFi-enabled microcontroller, good for learning register maps
- **STM32F103** — The "Blue Pill" board, ARM Cortex-M3, close relative of the Pico's RP2040
- **DHT22** — Temperature/humidity sensor used in v2.0 sensor logger
- **MPU-6050** — I2C accelerometer/gyro, directly connects to Day 19 (I2CPlayground)
- **SG90** — The servo used in v2.0 for PID-controlled pan-tilt tracking (Day 23)

## For Mentors

The Day 24 project asks students to chunk these PDFs, embed them, index them with chromadb, and build a Q&A agent. 3-5 datasheets is the right number — enough to test cross-document retrieval without overwhelming the embedding step.

If download links break, any version of these datasheets will work. The content matters more than the exact revision.
