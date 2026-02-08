# MagicBook (Book Spine Clock)

MagicBook turns a physical book into a clock: the e-paper spine changes over time to book titles that allude to the current time. It was built for a Raspberry Pi Zero 2 W and a Waveshare 7.5" tri-color (red/white/black) e-paper display.

## How to make your own and get it running

MakerWorld shell model: [ADD LINK HERE]

### Hardware

- Raspberry Pi Zero 2 W
- Waveshare 7.5" tri-color (red/white/black) e-paper display (480x800)
- MicroSD card, power, and SPI enabled

### Setup

1. Enable SPI on the Raspberry Pi (`raspi-config`).
2. Install dependencies:
   - `sudo apt-get install -y python3-pil`
3. Clone the repo and run the clock:
   - `python spine_clock.py`

If you want to test the panel first:

- `python waveshare_demo.py` (uses bundled Waveshare library in `lib/`)
- `python spine_demo.py` (renders a sample spine)

## How the books were selected

1. Candidate titles were pulled via API calls to Google Books and Open Library.
2. An LLM reviewed the candidate lists for time-related allusions.
3. Human curation selected the final set and scored results.

The curated list lives in `data/booknames_curated.csv`. The runtime schedule is generated into `data/booknames_curated_final.csv`.

## How the spines were made

1. LLM generates three spine design descriptions per book title (`prompts/spine_design_prompt.txt`).
2. LLM converts those descriptions into a custom JSON layout schema (`prompts/spine_json_prompt.txt`).
3. The schema is rendered into PNGs (`spine_renderer.py` → `pic/spines/`).
4. Human curation uses a local GUI to select the best spine (`curate_spines.py` → `pic/final/`).
5. Final images are converted into tri-color BMP layers for the display (`make_final_csv.py` → `pic/final_bmp/`).

## Project layout

- `spine_clock.py` - Main runtime loop; reads schedule CSV and updates the display
- `generate_spines.py` - End-to-end generation using OpenAI + rendering
- `curate_spines.py` - GUI for selecting the best spine per time slot
- `make_final_csv.py` - Builds the final CSV and BMP layers for the display
- `spine_renderer.py` - Renders JSON layouts into PNGs
- `image_to_epd_bmps.py` - Converts PNGs into black/red BMP layers
- `data/` - Curated titles, time schedules, and debug logs
- `pic/` - Generated PNGs, curated finals, and BMP output
- `prompts/` - LLM prompt templates
- `lib/` - Bundled Waveshare Python library

## Generate your own set (optional)

If you want to create a new set of spines and a fresh schedule:

1. Create or update `data/booknames_curated.csv`.
2. Generate candidate spines:
   - `python generate_spines.py`
3. Curate selections:
   - `python curate_spines.py`
4. Build the final schedule and BMP layers:
   - `python make_final_csv.py`

### Environment variables

Generation uses the OpenAI API and reads `.env` if present:

- `OPENAI_API_KEY`
- `OPENAI_TEXT_MODEL` (default: `gpt-5.2`)
- `OPENAI_LAYOUT_MODEL` (default: `gpt-5.2-codex`)
- `OPENAI_BASE_URL` (default: `https://api.openai.com`)

## Display notes

- The display is tri-color (black/red/white). The runtime uses full refresh periodically and fast refresh in-between.
- Image sizes: spines are rendered at 480x800, then split into BMP layers.

## Licensing

Code is licensed under the MIT License.

Dataset and images are licensed under Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0).

Individual book titles and author names are public facts, but the selection, structure, and time mapping are original to this project.

Commercial licensing inquiries: joaoperfig@gmail.com
