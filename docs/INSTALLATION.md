# Installation Guide

## 1. Prerequisites

- Python 3.10+ (tested on 3.12)
- pip
- (Optional, for real GeoTIFF stitching) a working GDAL install if
  `pip install rasterio` doesn't provide wheels for your platform

## 2. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `rasterio` fails to install (common on some Linux distros without
system GDAL), you can skip it - GeoTIFF tiles will still load via OpenCV,
just without geo-transform awareness:

```bash
pip install -r requirements.txt --no-deps rasterio  # or omit rasterio entirely
```

## 4. Place your trained model

Your existing model is already at:

```
models/oil_spill_attention_model.h5
```

If you retrain it, just overwrite this file - no code changes needed
(the path is read from `backend/config.py: MODEL_PATH`).

## 5. Run the application

```bash
python run.py
```

Visit **http://localhost:5000**.

Environment variables (all optional, see `backend/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `OIL_SPILL_DEBUG` | `1` | Flask debug mode |
| `OIL_SPILL_HOST` | `0.0.0.0` | Bind host |
| `OIL_SPILL_PORT` | `5000` | Bind port |
| `OIL_SPILL_SECRET_KEY` | dev key | Flask session secret - **set a real value in production** |
| `OIL_SPILL_LOG_LEVEL` | `INFO` | Logging verbosity |

## 6. Using the MOSDAC folder-scan flow

1. Drop one or more image tiles (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`)
   into `datasets/mosdac/incoming/`.
2. Open the app and click **Scan MOSDAC** (or visit `/mosdac/scan`).
3. The system stitches the tiles (if more than one), runs detection, logs
   the result, and moves the tiles to `datasets/mosdac/processed/`.

## 7. Production deployment notes

This ships with Flask's development server for simplicity. For a real
deployment:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 "backend.app:create_app()"
```

- Put it behind a reverse proxy (nginx/Caddy) for TLS termination.
- Set `OIL_SPILL_DEBUG=0` and a strong `OIL_SPILL_SECRET_KEY`.
- Mount `static/uploads/`, `outputs/`, and `datasets/mosdac/` on
  persistent storage (they're written to at runtime).
- Since TensorFlow model loading is lazy but cached as a singleton
  (`OilSpillPredictor.get_instance()`), the first request after a cold
  start will be slower (~1-2s) - consider a warm-up request in your
  deployment health check.
