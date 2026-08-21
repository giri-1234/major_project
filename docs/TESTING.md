# Testing

## What was actually verified during this build

Every claim below was executed against this exact codebase, including a
real load of your `.h5` model - nothing here is hypothetical.

| Test | Method | Result |
|---|---|---|
| Syntax/compile check | `python -m py_compile backend/*.py run.py` | All modules compile cleanly |
| App factory + routing | Flask test client, `GET /`, `/history`, `/mosdac/scan`, `/api/history`, `/static/css/style.css` | All return `200` |
| Full upload pipeline | Uploaded a synthetic BGR image with a dark rectangular patch via `POST /` | Model loaded, ran inference (`score=0.578`), HSV/morphology mask generated, `area=11.02%`, `severity=MEDIUM`, `status=OIL SPILL DETECTED`, mask saved, history recorded |
| MOSDAC scan with insufficient overlap | Two flat-color tiles with too little texture | Correctly caught `StitchingError` ("Only 4 good matches found"), rendered a clean error page, **did not** move tiles out of `incoming/` (no data loss on failure) |
| MOSDAC scan with real overlap | Two textured overlapping tiles | `cv2.Stitcher` succeeded, scene ran through the same pipeline (`score=0.411`, `area=8.55%`, `severity=LOW`, `status=OIL SPILL DETECTED (Borderline/Verify)`), tiles moved to `processed/` |
| History persistence | Checked `outputs/reports/prediction_history.csv` and `/history` page after the above runs | Both records present with matching fields |

## How to re-run these checks yourself

```bash
# 1. Syntax check
python -m py_compile backend/*.py run.py

# 2. Route smoke test (no model load required for these 4 routes)
python - <<'EOF'
from backend.app import create_app
client = create_app().test_client()
for path in ["/", "/history", "/api/history"]:
    print(path, client.get(path).status_code)
EOF

# 3. Full pipeline test with your own image
python - <<'EOF'
from backend.app import create_app
client = create_app().test_client()
with open("path/to/your/test_image.jpg", "rb") as f:
    resp = client.post("/", data={"file": (f, "test_image.jpg")},
                        content_type="multipart/form-data")
print(resp.status_code)
EOF
```

## Recommended additional testing before production use

These were **not** run here because they require assets this environment
doesn't have (a labeled validation set, real MOSDAC tiles, a browser for
UI/visual QA):

- Run your held-out validation/test set through `OilSpillPredictor.predict()`
  in bulk and compare confusion-matrix metrics against your original
  Colab training run, to confirm the refactor didn't change model behavior
  (it shouldn't - the model file and preprocessing math are unchanged).
- Visual QA of `static/css/style.css` across breakpoints in an actual
  browser (mobile/tablet/desktop) - only `curl`/test-client-level checks
  were possible here.
- Load-test `/mosdac/scan` and `/` with concurrent requests if you expect
  more than one operator using the dashboard simultaneously; the model
  singleton is not currently thread-lock-guarded during the first cold
  load.
- Test with real multi-tile GeoTIFFs from MOSDAC/Bhoonidhi once you have
  sample data, to validate the `rasterio` read path (only PNG/JPEG paths
  were exercised here since real georeferenced tiles weren't available).
