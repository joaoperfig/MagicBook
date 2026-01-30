# Book Cover Clock

Raspberry Pi Zero 2 W with a Waveshare 7.5-inch tri-color (red/white/black) e‑Paper HAT.

This project turns a physical book into a clock: the cover image changes over time by selecting book titles that represent the current time.

## Dependencies

- Python 3
- Waveshare e‑Paper Python library (bundled in `lib/`)
- Pillow (PIL)

## Setup

1. Ensure SPI is enabled on the Raspberry Pi.
2. Install Python dependencies:
   - `sudo apt-get install -y python3-pil`
3. Run the demo script:
   - `python waveshare_demo.py`

Notes:
- The demo uses assets in `pic/` and the local Waveshare library in `lib/`.
- Add any secrets to `secrets.env` (this file is ignored by git).
