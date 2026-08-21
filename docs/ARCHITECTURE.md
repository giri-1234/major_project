# Architecture

## 1. System architecture

```
                        ┌─────────────────────────────┐
                        │        Web Browser           │
                        │  (upload / scan / history)   │
                        └───────────────┬───────────────┘
                                        │ HTTP
                        ┌───────────────▼───────────────┐
                        │        Flask Application        │
                        │      backend/app.py (factory)   │
                        │      backend/routes.py (views)   │
                        └───────────────┬───────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────▼────────┐          ┌───────────▼───────────┐        ┌──────────▼─────────┐
│  mosdac.py       │          │   stitching.py          │        │  preprocessing.py    │
│  MosdacIngestor  │─tiles──▶│   ImageStitcher fns     │─scene─▶│  pipeline stages     │
│  (folder scan;   │          │   cv2.Stitcher +        │        │  resize/denoise/     │
│  API stub ready) │          │   ORB/homography        │        │  blur/CLAHE/HSV/     │
└─────────────────┘          │   fallback               │        │  morphology/ROI      │
                              └─────────────────────────┘        └──────────┬─────────┘
                                                                            │
                                                                 ┌──────────▼─────────┐
                                                                 │   prediction.py       │
                                                                 │   OilSpillPredictor   │
                                                                 │   - loads .h5 model   │
                                                                 │   - CNN confidence    │
                                                                 │   - area estimation   │
                                                                 │   - decision logic    │
                                                                 └──────────┬─────────┘
                                                                            │
                              ┌─────────────────────────────────────────────┼───────────────┐
                              │                                             │               │
                    ┌─────────▼─────────┐                        ┌──────────▼──────┐   ┌────▼────┐
                    │   utils.py          │                        │  static/uploads/  │   │ Dashboard│
                    │  - logging           │                        │  masks, stitched  │   │ templates│
                    │  - history CSV/JSON  │                        │  scenes, originals│   │ (Jinja2) │
                    │  - severity + alerts │                        └──────────────────┘   └─────────┘
                    └─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  outputs/           │
                    │  logs/system.log    │
                    │  reports/*.csv,json │
                    └────────────────────┘
```

## 2. Data flow

```
 MOSDAC incoming/            Manual upload
       │                          │
       ▼                          │
 [MosdacIngestor.scan]            │
       │                          │
       ▼                          │
 2+ tiles? ──yes──▶ [stitching.stitch_tiles]
       │no                        │
       ▼                          ▼
 single tile ─────────────▶  one scene image
                                   │
                                   ▼
                    [preprocessing.run_preprocessing_pipeline]
                    resize → denoise → Gaussian blur → CLAHE →
                    HSV convert → threshold → morphology open/close →
                    ROI crop
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                               ▼
        [prediction._predict_raw_score]   [prediction._estimate_area]
        CNN sigmoid confidence              oil-pixel % of ROI
                    │                               │
                    └──────────────┬────────────────┘
                                   ▼
                    [prediction._decide] double-gate logic
                    → status, color, severity
                                   │
                    ┌──────────────┼────────────────┐
                    ▼              ▼                ▼
              save mask     utils.build_alert   utils.append_history_record
              (if spill)    (LOW/MEDIUM/HIGH)   (CSV + JSON)
                    │              │                │
                    └──────────────┴────────────────┘
                                   ▼
                        render results.html
                        (dashboard: original, stitched,
                         mask, metrics, alert, timestamp)
```

## 3. Class diagram (core objects)

```
┌────────────────────────────┐
│  OilSpillPredictor           │
├────────────────────────────┤
│ - _model: keras.Model         │
│ - _instance: OilSpillPredictor│ (singleton)
├────────────────────────────┤
│ + get_instance() -> Self      │
│ + predict(filepath) -> PredictionResult │
│ - _predict_raw_score(filepath) -> float │
│ - _estimate_area(result) -> float       │
│ - _decide(score, area) -> tuple         │
│ - _save_mask(filepath, mask) -> str     │
└────────────────────────────┘
             │ uses
             ▼
┌────────────────────────────┐        ┌───────────────────────────┐
│  PredictionResult (dataclass)│       │  PreprocessedResult (dataclass)│
├────────────────────────────┤        ├───────────────────────────┤
│ status, color, confidence,   │       │ original, resized, denoised,   │
│ area_pct, severity,           │       │ blurred, contrast_enhanced,    │
│ mask_filename,                │       │ hsv, oil_mask_raw,             │
│ processing_time_ms            │       │ oil_mask_clean, roi_mask,      │
└────────────────────────────┘        │ display_preprocessed           │
                                       └───────────────────────────┘

┌────────────────────────────┐        ┌───────────────────────────┐
│  MosdacIngestor              │        │  MosdacScene (dataclass)     │
├────────────────────────────┤        ├───────────────────────────┤
│ - incoming_dir, processed_dir│        │ scene_id, tile_paths,         │
├────────────────────────────┤        │ acquired_at                   │
│ + scan_incoming_folder()      │───▶  └───────────────────────────┘
│ + fetch_new_images() -> Scene │
│ + mark_processed(scene)       │
│ + fetch_from_api(...) [stub]  │
└────────────────────────────┘

┌────────────────────────────┐        ┌───────────────────────────┐
│  StitchResult (dataclass)    │        │  PredictionRecord (dataclass)│
├────────────────────────────┤        ├───────────────────────────┤
│ stitched_image, tile_count,   │        │ record_id, timestamp,         │
│ method_used, output_path      │        │ image_source, original_image, │
└────────────────────────────┘        │ stitched_image, mask_image,   │
                                       │ status, confidence, area_pct, │
                                       │ severity, processing_time_ms  │
                                       └───────────────────────────┘
```

## 4. Sequence diagram — MOSDAC scan request

```
Browser        routes.py       MosdacIngestor    stitching.py    preprocessing.py   OilSpillPredictor   utils.py
   │  GET /mosdac/scan │               │                │                │                  │              │
   │──────────────────▶│               │                │                │                  │              │
   │                    │─fetch_new_images()▶│           │                │                  │              │
   │                    │◀── MosdacScene ────│           │                │                  │              │
   │                    │─stitch_tiles(tile_paths)───────▶│                │                  │              │
   │                    │◀── StitchResult ────────────────│                │                  │              │
   │                    │──predict(stitched_path)────────────────────────────────────────────▶│              │
   │                    │                │                │─run_preprocessing_pipeline()──▶│                  │              │
   │                    │                │                │◀── PreprocessedResult ─────────│                  │              │
   │                    │                │                │                │──_predict_raw_score()           │
   │                    │                │                │                │──_estimate_area()                │
   │                    │                │                │                │──_decide()                       │
   │                    │◀── PredictionResult ─────────────────────────────────────────────────│              │
   │                    │──build_alert(status, severity, area)───────────────────────────────────────────────▶│
   │                    │◀── alert dict / None ──────────────────────────────────────────────────────────────│
   │                    │──append_history_record(record)────────────────────────────────────────────────────▶│
   │                    │──mark_processed(scene)──▶│      │                │                  │              │
   │                    │◀── moved to processed/──│       │                │                  │              │
   │◀── render results.html ──│               │                │                │                  │              │
```
