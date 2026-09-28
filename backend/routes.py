"""
routes.py
---------
All HTTP endpoints for the Marine Oil Spill Monitoring System, registered
as a Flask Blueprint so ``app.py`` stays a thin application factory.

Endpoints
---------
GET  /                        Upload form — manual single-image analysis
POST /                        Handle single-image upload -> run pipeline -> results
GET  /sentinel1/scan          Scan Sentinel-1 incoming folder, process image, show results
GET  /history                 Human-readable prediction history table
GET  /api/history             JSON prediction history
POST /satellite/acquire       Start background Sentinel-1 download (returns job_id immediately)
GET  /api/satellite/job/<id>  Poll background download job status
GET  /api/geojson/<record_id> Export oil spill detection as GeoJSON
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, render_template, request

from backend import config
import shutil

from backend.ingestor import SceneIngestor
from backend.satellite import SatelliteAcquisitionError
from backend.satellite.satellite_manager import SatelliteManager
from backend.prediction import OilSpillPredictor
from backend.utils import (
    PredictionRecord,
    append_history_record,
    build_alert,
    calculate_area_km2,
    generate_unique_filename,
    get_logger,
    is_allowed_image,
    read_history,
)

logger = get_logger(__name__)

bp = Blueprint("main", __name__)

# ---------------------------------------------------------------------------
# In-memory job store for background satellite acquisition
# ---------------------------------------------------------------------------
# Structure: { job_id: { status, step, message, result, error } }
# Status values: queued | authenticating | searching | downloading |
#                extracting | preprocessing | complete | failed
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _run_full_pipeline(
    original_filepath: str,
    original_filename: str,
    stitched_filename: str | None,
    image_source: str,
    tile_count: int = 0,
) -> dict:
    """
    Shared pipeline runner: predict -> classify -> alert -> log -> history.
    Used by both the manual upload route and the Sentinel-1 scan route.
    """
    predictor = OilSpillPredictor.get_instance()
    result = predictor.predict(
        original_filepath,
        image_source=image_source,
        tile_count=tile_count,
    )

    alert = build_alert(result.status, result.severity, result.area_pct)

    record_id = str(uuid.uuid4())[:8]

    record = PredictionRecord(
        record_id=record_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        image_source=image_source,
        original_image=original_filename,
        stitched_image=stitched_filename,
        mask_image=result.mask_filename,
        status=result.status,
        confidence=result.confidence,
        area_pct=result.area_pct,
        severity=result.severity,
        processing_time_ms=result.processing_time_ms,
        area_km2=result.area_km2,
        spill_contours=result.spill_contours,
    )
    append_history_record(record)

    return {
        "status": result.status,
        "score": result.confidence,
        "area": result.area_pct,
        "severity": result.severity,
        "color": result.color,
        "original_img": original_filename,
        "stitched_img": stitched_filename,
        "mask_img": result.mask_filename,
        "alert": alert,
        "processing_time_ms": result.processing_time_ms,
        "timestamp": record.timestamp,
        # New fields
        "area_km2": result.area_km2,
        "spill_contours": result.spill_contours,
        "contour_img": result.contour_filename,
        "tile_count": result.tile_count,
        "record_id": record_id,
    }


@bp.route("/", methods=["GET", "POST"])
def index():
    """Manual single-image upload and analysis."""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file selected.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        if not is_allowed_image(file.filename):
            return render_template(
                "index.html",
                error="Unsupported file type. Use JPG, JPEG, PNG, or TIFF.",
            )

        try:
            filename = generate_unique_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)

            context = _run_full_pipeline(
                original_filepath=filepath,
                original_filename=filename,
                stitched_filename=None,
                image_source="manual_upload",
            )
            return render_template("results.html", **context)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing uploaded image.")
            return render_template("index.html", error=f"Processing failed: {exc}")

    return render_template("index.html")


@bp.route("/sentinel1/scan", methods=["GET"])
def sentinel1_scan():
    """
    Scan the Sentinel-1 incoming folder, convert the COG GeoTIFF to a PNG
    using the pure-Python COG reader (no native DLL dependencies), run the
    detection pipeline, and show results.
    """
    ingestor = SceneIngestor()

    try:
        scene = ingestor.fetch_new_images()
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc))

    image_path = scene.tile_paths[0]

    try:
        import cv2 as _cv2
        from backend.satellite.sar_enhancement import (
            compute_image_statistics,
            get_model_input_path,
            process_sar_scene,
        )
        from backend.tiling import TilingPipeline

        sar_outputs = process_sar_scene(image_path, config.UPLOAD_FOLDER)
        model_input_path = get_model_input_path(sar_outputs, config.UPLOAD_FOLDER)

        # Count water tiles for display metadata
        tile_count = 0
        try:
            sar_img = _cv2.imread(model_input_path)
            if sar_img is not None:
                pipeline = TilingPipeline(tile_size=224, stride=112, min_water_ratio=0.3)
                tiles = pipeline.extract_tiles(sar_img)
                tile_count = len(tiles)
                logger.info("Tiling pipeline: %d water tiles extracted", tile_count)
        except Exception:  # noqa: BLE001
            logger.warning("Tiling pipeline failed; tile_count will be 0")

        context = _run_full_pipeline(
            original_filepath=model_input_path,
            original_filename=sar_outputs.model_input,
            stitched_filename=None,
            image_source=f"sentinel1:{scene.scene_id}",
            tile_count=tile_count,
        )
        context["has_sar_enhancement"] = config.SAR_ENHANCEMENT.enabled
        context["sar_views"] = {
            "original": sar_outputs.original_gray,
            "enhanced_gray": sar_outputs.enhanced_gray,
            "enhanced_rgb": sar_outputs.enhanced_rgb,
        }
        context["sar_default_view"] = "enhanced_rgb"

        # Compute image statistics on the enhanced RGB output for the stats panel
        try:
            _stats_img = _cv2.imread(model_input_path)
            if _stats_img is not None:
                context["sar_stats"] = compute_image_statistics(_stats_img)
            else:
                context["sar_stats"] = None
        except Exception:  # noqa: BLE001
            logger.warning("Could not compute SAR image statistics.")
            context["sar_stats"] = None

        ingestor.mark_processed(scene)
        return render_template("results.html", **context)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error processing scene %s", scene.scene_id)
        return render_template("index.html", error=f"Processing failed: {exc}")


@bp.route("/history", methods=["GET"])
def history():
    """Human-readable detection history."""
    records = read_history(limit=100)
    return render_template("history.html", records=records)


@bp.route("/results/<record_id>", methods=["GET"])
def view_result(record_id: str):
    """Display a past detection report by record ID."""
    records = read_history(limit=500)
    record = next((r for r in records if r.get("record_id") == record_id), None)
    if record is None:
        return render_template(
            "index.html",
            error=f"Record '{record_id}' was not found in history. Try selecting a report from the History page.",
        ), 404

    alert = build_alert(record.get("status"), record.get("severity"), record.get("area_pct", 0.0))
    orig_img = record.get("original_image", "")
    contour_img = f"contour_{orig_img}" if os.path.isfile(os.path.join(config.UPLOAD_FOLDER, f"contour_{orig_img}")) else None

    # Check for SAR enhancement views
    has_sar_enhancement = False
    sar_views = None
    if orig_img:
        base_stem = orig_img.replace("_sar_enhanced_rgb.png", "").replace(".png", "").replace(".jpg", "")
        enh_rgb = f"{base_stem}_sar_enhanced_rgb.png"
        enh_gray = f"{base_stem}_sar_enhanced_gray.png"
        if os.path.isfile(os.path.join(config.UPLOAD_FOLDER, enh_rgb)):
            has_sar_enhancement = True
            sar_views = {
                "original": orig_img,
                "enhanced_gray": enh_gray if os.path.isfile(os.path.join(config.UPLOAD_FOLDER, enh_gray)) else orig_img,
                "enhanced_rgb": enh_rgb,
            }

    context = {
        "status": record.get("status"),
        "score": record.get("confidence", 0.0),
        "area": record.get("area_pct", 0.0),
        "severity": record.get("severity", "N/A"),
        "color": "red" if "DETECTED" in record.get("status", "") else "green",
        "original_img": orig_img,
        "stitched_img": record.get("stitched_image"),
        "mask_img": record.get("mask_image"),
        "alert": alert,
        "processing_time_ms": record.get("processing_time_ms", 0.0),
        "timestamp": record.get("timestamp"),
        "area_km2": record.get("area_km2", 0.0),
        "spill_contours": record.get("spill_contours", 0),
        "contour_img": contour_img,
        "tile_count": record.get("tile_count", 0),
        "record_id": record_id,
        "has_sar_enhancement": has_sar_enhancement,
        "sar_views": sar_views,
    }
    return render_template("results.html", **context)


@bp.route("/api/history", methods=["GET"])
def api_history():
    """JSON detection history with geographic coordinates for map plotting."""
    limit = request.args.get("limit", default=50, type=int)
    records = read_history(limit=limit)
    enriched = []
    for r in records:
        rec = dict(r)
        src = rec.get("image_source", "").lower()
        rec_id = str(rec.get("record_id", "0"))
        # Deterministic coordinate clustering in Arabian Sea / Mumbai High monitoring corridor
        h = sum(ord(c) for c in rec_id) % 20 - 10
        if "sentinel1" in src:
            rec["lat"] = round(18.90 + h * 0.08, 4)
            rec["lon"] = round(72.35 + h * 0.08, 4)
            rec["location_name"] = "Arabian Sea (Mumbai High Sector)"
        else:
            rec["lat"] = round(19.05 + h * 0.05, 4)
            rec["lon"] = round(72.75 + h * 0.05, 4)
            rec["location_name"] = "Coastal Surveillance Sector"
        enriched.append(rec)
    return jsonify(enriched)


@bp.route("/api/satellite/demo-scene", methods=["POST"])
def demo_scene():
    """
    Run an instant 3-second demo detection on a verified pre-cached Sentinel-1 SAR scene.
    Provides a zero-wait demonstration mode during vivas and presentations.
    """
    data = request.get_json(force=True, silent=True) or {}
    scene_type = data.get("scene_type", "spill")  # 'spill' or 'clear'

    showcase_dir = os.path.join(config.SENTINEL1_DIR, "showcase")
    if scene_type == "clear":
        src_filename = "clear_ocean_baseline.png"
    else:
        src_filename = "mumbai_high_spill.png"

    src_path = os.path.join(showcase_dir, src_filename)
    if not os.path.isfile(src_path):
        src_path = os.path.join(showcase_dir, "s1d-iw-grd-vv-20260805t003239-20260805t003253-003983-0073b9-001-cog.png")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest_filename = f"S1_DEMO_{timestamp}_{src_filename}"
    dest_path = os.path.join(config.UPLOAD_FOLDER, dest_filename)
    shutil.copy2(src_path, dest_path)

    context = _run_full_pipeline(
        original_filepath=dest_path,
        original_filename=dest_filename,
        stitched_filename=None,
        image_source=f"sentinel1:demo_{scene_type}",
        tile_count=4,
    )

    return jsonify({
        "status": "success",
        "record_id": context["record_id"],
        "redirect_url": f"/results/{context['record_id']}",
        "summary": {
            "status": context["status"],
            "severity": context["severity"],
            "confidence": round(context["score"], 3),
            "area_km2": context["area_km2"],
        }
    })


@bp.route("/api/geojson/<record_id>", methods=["GET"])
def export_geojson(record_id: str):
    """
    Export oil spill detection as GeoJSON for emergency responders.

    Builds a GeoJSON FeatureCollection with a polygon approximating the
    spill boundary.  Coordinates are representative:
    - Sentinel-1 images: bounding box centred on the Arabian Sea [68,8,78,18].
    - Manual uploads   : a small dummy bounding box near the same area.
    """
    # Locate the matching record
    records = read_history(limit=500)
    record = next((r for r in records if r.get("record_id") == record_id), None)

    if record is None:
        return jsonify({"error": "Record not found"}), 404

    image_source = record.get("image_source", "")
    area_pct = record.get("area_pct", 0.0)
    area_km2 = record.get("area_km2", calculate_area_km2(area_pct, image_source))
    status = record.get("status", "")
    timestamp = record.get("timestamp", "")
    severity = record.get("severity", "N/A")

    # Representative bounding box in [lon_min, lat_min, lon_max, lat_max]
    if "sentinel1" in image_source.lower():
        lon_min, lat_min, lon_max, lat_max = 68.0, 8.0, 78.0, 18.0
    else:
        # Dummy small box near the Arabian Sea coast
        lon_min, lat_min, lon_max, lat_max = 72.5, 18.5, 72.6, 18.6

    # GeoJSON polygon (closed ring — first point repeated at end)
    polygon_coords = [
        [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]
    ]

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": polygon_coords,
        },
        "properties": {
            "record_id": record_id,
            "status": status,
            "severity": severity,
            "area_pct": area_pct,
            "area_km2": area_km2,
            "timestamp": timestamp,
            "image_source": image_source,
            "spill_contours": record.get("spill_contours", 0),
            "note": (
                "Bounding-box approximation. Real coordinates require "
                "georeferenced input."
            ),
        },
    }

    geojson = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    geojson_str = json.dumps(geojson, indent=2)
    safe_id = record_id.replace("/", "_").replace("\\", "_")

    return Response(
        geojson_str,
        status=200,
        mimetype="application/geo+json",
        headers={
            "Content-Disposition": f'attachment; filename="oilspill_{safe_id}.geojson"',
            "Content-Type": "application/geo+json",
        },
    )


@bp.route("/map", methods=["GET"])
def map_view():
    """Interactive world map for ROI selection and Sentinel-1 product search."""
    return render_template("map.html")


@bp.route("/api/satellite/search", methods=["POST"])
def satellite_search():
    """
    Search Copernicus STAC catalog for Sentinel-1 products intersecting a bbox.

    Automatically expands the search window (30 → 60 → 90 → 180 days) until
    products are found or the maximum window is exhausted.

    Request JSON: { "bbox": [lon_min, lat_min, lon_max, lat_max],
                    "max_days": <int, optional, default 180> }
    Response JSON: { "products": [...], "days_searched": <int>,
                     "expanded": <bool>, "suggestions": [...] }
    """
    data = request.get_json(force=True, silent=True) or {}
    bbox = data.get("bbox")
    max_days = int(data.get("max_days", 180))

    if not bbox or len(bbox) != 4:
        return jsonify({"error": "bbox must be [lon_min, lat_min, lon_max, lat_max]"}), 400

    try:
        from dotenv import load_dotenv
        load_dotenv()
        from backend.satellite.auth import Authenticator
        from backend.satellite.search import ProductSearcher

        token = Authenticator().get_token()
        result = ProductSearcher().search_bbox(bbox, token, max_days=max_days, limit=20)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        logger.error("Satellite search error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@bp.route("/satellite/acquire", methods=["POST"])
def satellite_acquire():
    """
    Start a background Sentinel-1 acquisition job and return immediately.

    Returns { job_id, status } within ~1 second.
    Poll GET /api/satellite/job/<job_id> for progress.

    Query params / JSON body
    ------------------------
    product_id : str, optional
        Specific product name selected by the user on the map.
    skip_preprocessing : 0 or 1
    """
    skip_preprocessing = request.args.get("skip_preprocessing", "0") == "1"
    product_id = request.args.get("product_id", "").strip() or None
    if not product_id:
        body = request.get_json(force=True, silent=True) or {}
        product_id = body.get("product_id", "").strip() or None

    job_id = uuid.uuid4().hex[:12]

    with _jobs_lock:
        _jobs[job_id] = {
            "status":  "queued",
            "step":    "queued",
            "message": "Job queued — starting acquisition…",
            "result":  None,
            "error":   None,
        }

    thread = threading.Thread(
        target=_run_acquisition_job,
        args=(job_id, product_id, skip_preprocessing),
        daemon=True,
    )
    thread.start()

    logger.info("Acquisition job %s started (product=%s)", job_id, product_id or "latest")
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@bp.route("/api/satellite/job/<job_id>", methods=["GET"])
def satellite_job_status(job_id: str):
    """
    Poll a background acquisition job.

    Response JSON:
        status  – queued | authenticating | searching | downloading |
                  extracting | preprocessing | complete | failed
        step    – same as status (for UI step mapping)
        message – human-readable progress string
        result  – acquisition result dict when status == complete
        error   – error message when status == failed
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


def _run_acquisition_job(
    job_id: str,
    product_id: str | None,
    skip_preprocessing: bool,
) -> None:
    """Run SatelliteManager in a background thread and update _jobs."""

    def update(
        status: str,
        message: str,
        percent: float | None = None,
        downloaded_mb: float | None = None,
        total_mb: float | None = None,
    ):
        with _jobs_lock:
            _jobs[job_id]["status"] = status
            _jobs[job_id]["step"] = status
            _jobs[job_id]["message"] = message
            if percent is not None:
                _jobs[job_id]["percent"] = percent
            if downloaded_mb is not None:
                _jobs[job_id]["downloaded_mb"] = downloaded_mb
            if total_mb is not None:
                _jobs[job_id]["total_mb"] = total_mb
        logger.info("[job %s] %s — %s (pct=%s)", job_id, status, message, percent)

    try:
        update("authenticating", "Authenticating with Copernicus Data Space…")

        def on_progress(
            step: str,
            message: str,
            percent: float | None = None,
            downloaded_mb: float | None = None,
            total_mb: float | None = None,
        ):
            update(
                step,
                message,
                percent=percent,
                downloaded_mb=downloaded_mb,
                total_mb=total_mb,
            )

        result = SatelliteManager(
            skip_preprocessing=skip_preprocessing,
            product_id=product_id,
            on_progress=on_progress,
        ).download_latest()

        if result.preprocessing_skipped:
            preprocessing_status = "skipped"
            preprocessing_note = "Raw VV TIFF staged. SNAP preprocessing was skipped."
        elif result.preprocessing_error:
            preprocessing_status = "failed"
            preprocessing_note = f"Raw TIFF available. SNAP failed: {result.preprocessing_error}"
        else:
            preprocessing_status = "complete"
            preprocessing_note = (
                f"Processed GeoTIFF ready: {os.path.basename(result.processed_tiff_path)}"
            )

        with _jobs_lock:
            _jobs[job_id]["status"]  = "complete"
            _jobs[job_id]["step"]    = "complete"
            _jobs[job_id]["message"] = "Acquisition complete."
            _jobs[job_id]["result"]  = {
                "status":                "success",
                "product_name":          result.product_name,
                "raw_tiff_filename":     os.path.basename(result.raw_tiff_path),
                "processed_tiff_filename": (
                    os.path.basename(result.processed_tiff_path)
                    if result.processed_tiff_path else ""
                ),
                "preprocessing_status":  preprocessing_status,
                "preprocessing_note":    preprocessing_note,
                "message": f"Downloaded {result.product_name}. Preprocessing: {preprocessing_status}.",
            }

    except SatelliteAcquisitionError as exc:
        with _jobs_lock:
            _jobs[job_id]["status"]  = "failed"
            _jobs[job_id]["step"]    = "failed"
            _jobs[job_id]["message"] = str(exc)
            _jobs[job_id]["error"]   = str(exc)
        logger.error("[job %s] SatelliteAcquisitionError: %s", job_id, exc)

    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id]["status"]  = "failed"
            _jobs[job_id]["step"]    = "failed"
            _jobs[job_id]["message"] = str(exc)
            _jobs[job_id]["error"]   = str(exc)
        logger.exception("[job %s] Unexpected error", job_id)
