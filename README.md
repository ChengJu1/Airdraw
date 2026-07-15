# Airdraw

Air-drawn sketch capture with the Skywriter (MGC3130), trajectory preprocessing, and AI reconstruction. Includes a browser demo (`index.html`), a Raspberry Pi exhibition app (`draw_app.py`), and an offline processing pipeline.

```
raw (t, x, y, z) → pen up/down detection → denoise → segment/resample/normalize → stroke-3 → LLM reconstruction
```

## Web demo

Open `index.html` in a browser, or see `demo.mp4` and the `samples/` folder for example sketch-to-art outputs.

## Pipeline modules

| File | Purpose |
|------|------|
| `trajectory.py` | Trajectory data structure + CSV I/O |
| `pen_state.py`  | Pen up/down detection (dual-threshold hysteresis on Z height; supports explicit pen column) |
| `denoise.py`    | Outlier removal + 1€ filter smoothing (real-time friendly) |
| `extract.py`    | Segmentation, equal arc-length resampling, normalization, stroke-3 conversion |
| `visualize.py`  | Three-panel plot: raw / denoised+pen-down / extracted strokes |
| `demo.py`       | Run the full pipeline on synthetic data, or load a real CSV |
| `rt_filters.py` | Real-time filter chain (median / spike gate / 1€ / z hysteresis pen up/down), shared by the Raspberry Pi and the offline side |
| `mgc3130.py`    | MGC3130 sensor I2C reader, shared by the three Raspberry Pi scripts (copy along with the scripts when deploying) |
| `draw_app.py`     | Fullscreen Pygame exhibition app on the Pi |
| `web_capture.py`  | Flask app for live drawing in a browser |
| `reconstruct_llm.py`| LLM recognition + text-to-image reconstruction demo |

## Pen up/down scheme

Default is **dual-threshold hysteresis on Z height**: hand low (small z) = pen down, hand high (large z) = pen up; down threshold `0.30`, up threshold `0.45`, and the in-between band keeps the previous state to avoid jitter. Thresholds can be adjusted in `pen_state.pen_from_z()`, or set `z_is_height=False`.
If the capture already carries an explicit pen-down flag, add a `pen` column to the CSV (1=down, 0=up); it takes priority.

## Usage

```bash
pip install -r requirements.txt      # PC side (offline pipeline)
# Raspberry Pi side: pip install -r requirements_pi.txt (see deployment notes at the top of that file)

# 1) Run with synthetic data (generates data/synthetic.csv and results under out/)
python demo.py

# 2) Run with real capture data (CSV needs at least t,x,y; optional z,pen)
python demo.py data/your_capture.csv
```

Outputs:
- `out/<name>_pipeline.png`: three-panel plot of denoising and extraction results
- `out/<name>_stroke3.npy`: stroke-3 sequence (fed to Sketch-RNN in the next stage)

## CSV format

| Column | Meaning |
|----|------|
| `t` | timestamp (seconds), defaults to frame index if absent |
| `x`, `y` | normalized coordinates 0~1 (may be empty when the hand is out of range) |
| `z` | hand height above the panel 0~1 (for pen up/down), optional |
| `pen` | explicit pen-down flag 1/0, optional (takes priority over z) |

## Next step

The stroke-3 sequence is fed into Sketch-RNN or an LLM pipeline for the **reconstruction** stage.
