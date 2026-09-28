/**
 * map.js — Operational Maritime GIS for Sentinel-1 Oil Spill Surveillance
 *
 * Capabilities:
 *  - Multi-layer basemap switcher (ESRI Dark Gray Canvas, ESRI World Satellite, OpenStreetMap)
 *  - Smart 1-click presets that pan AND auto-draw guaranteed bounding boxes with km² calculation
 *  - Orbital swath footprint rendering: displays exact satellite coverage polygons on the ocean
 *  - Historical incident markers: plots past detection records with severity badges, thumbnails & links
 *  - Dual operational mode: Instant 3-second emergency demo stream OR full background raw downlink
 */
(function () {
    "use strict";

    // ------------------------------------------------------------------
    // 1. Basemap Layers & Map Initialisation
    // ------------------------------------------------------------------
    var darkBase = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        {
            attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
            maxZoom: 18,
        }
    );

    // darkRef removed — it added ship/port marker dots which cluttered the marine view
    var darkLayer = L.layerGroup([darkBase]);

    var satelliteLayer = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            maxZoom: 18,
        }
    );

    var osmLayer = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            maxZoom: 18,
        }
    );

    // Initialize map with ESRI Dark Gray Canvas by default (matches marine sci-fi UI)
    var map = L.map("map", {
        center: [18.9, 72.8],
        zoom: 6,
        layers: [darkLayer],
    });

    // Layer Groups
    var drawnItems = new L.FeatureGroup().addTo(map);
    var footprintsGroup = new L.FeatureGroup().addTo(map);
    var incidentsGroup = new L.FeatureGroup(); // Hidden by default; toggled via 'Past Incidents' checkbox

    var baseMaps = {
        "Dark Canvas": darkLayer,
        "Satellite Imagery": satelliteLayer,
        "OpenStreetMap": osmLayer,
    };

    var overlayMaps = {
        "Selected ROI": drawnItems,
        "Sentinel-1 Swaths": footprintsGroup,
        "Oil Spill Incidents": incidentsGroup,
    };

    L.control.layers(baseMaps, overlayMaps, { position: "topright" }).addTo(map);

    // Leaflet Draw Control
    var drawControl = new L.Control.Draw({
        position: "topleft",
        draw: {
            rectangle: {
                shapeOptions: {
                    color: "#37e0c4",
                    weight: 2,
                    fillColor: "#37e0c4",
                    fillOpacity: 0.15,
                },
                showArea: true,
            },
            polygon: false,
            polyline: false,
            circle: false,
            circlemarker: false,
            marker: false,
        },
        edit: { featureGroup: drawnItems, remove: true },
    });
    map.addControl(drawControl);

    var currentBounds = null;
    var selectedProduct = null;
    var productLayers = {}; // product_id -> L.GeoJSON layer

    // ------------------------------------------------------------------
    // 2. Draw Events
    // ------------------------------------------------------------------
    map.on(L.Draw.Event.CREATED, function (e) {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        currentBounds = e.layer.getBounds();
        showROI(currentBounds);
        document.getElementById("btn-draw").classList.remove("active");
        document.getElementById("btn-draw").textContent = "✏ Draw Rectangle";
    });

    map.on(L.Draw.Event.DELETED, function () {
        currentBounds = null;
        document.getElementById("roi-placeholder").style.display = "block";
        document.getElementById("roi-details").style.display = "none";
        hide("results-panel");
        hide("download-panel");
        footprintsGroup.clearLayers();
    });

    map.on(L.Draw.Event.EDITED, function () {
        drawnItems.eachLayer(function (l) {
            currentBounds = l.getBounds();
            showROI(currentBounds);
        });
    });

    // ------------------------------------------------------------------
    // 3. Toolbar Buttons & Smart Presets
    // ------------------------------------------------------------------
    document.getElementById("btn-draw").addEventListener("click", function () {
        var h = new L.Draw.Rectangle(map, drawControl.options.draw.rectangle);
        h.enable();
        this.classList.add("active");
        this.textContent = "✏ Click & drag on map…";
    });

    document.getElementById("btn-clear").addEventListener("click", function () {
        drawnItems.clearLayers();
        footprintsGroup.clearLayers();
        currentBounds = null;
        document.getElementById("roi-placeholder").style.display = "block";
        document.getElementById("roi-details").style.display = "none";
        hide("results-panel");
        hide("download-panel");
    });

    document.getElementById("btn-use-view").addEventListener("click", function () {
        drawnItems.clearLayers();
        currentBounds = map.getBounds();
        L.rectangle(currentBounds, {
            color: "#37e0c4",
            weight: 2,
            fillColor: "#37e0c4",
            fillOpacity: 0.15,
        }).addTo(drawnItems);
        showROI(currentBounds);
    });

    // Smart 1-Click Presets: Pans AND Automatically Draws Guaranteed Bounding Box
    document.querySelectorAll(".preset-btn").forEach(function (b) {
        b.addEventListener("click", function () {
            document.querySelectorAll(".preset-btn").forEach(function (btn) {
                btn.classList.remove("active");
            });
            b.classList.add("active");

            var bboxStr = b.dataset.bbox;
            if (bboxStr) {
                try {
                    var bbox = JSON.parse(bboxStr); // [lon_min, lat_min, lon_max, lat_max]
                    var bounds = L.latLngBounds([
                        [bbox[1], bbox[0]],
                        [bbox[3], bbox[2]],
                    ]);
                    drawnItems.clearLayers();
                    currentBounds = bounds;
                    L.rectangle(bounds, {
                        color: "#37e0c4",
                        weight: 2,
                        fillColor: "#37e0c4",
                        fillOpacity: 0.15,
                    }).addTo(drawnItems);
                    showROI(bounds);
                    map.fitBounds(bounds, { padding: [30, 30] });
                    return;
                } catch (err) {
                    console.error("Error parsing preset bbox:", err);
                }
            }

            // Fallback
            map.setView(
                [parseFloat(b.dataset.lat), parseFloat(b.dataset.lng)],
                parseInt(b.dataset.zoom)
            );
        });
    });

    // ------------------------------------------------------------------
    // 4. ROI Display Helpers
    // ------------------------------------------------------------------
    function showROI(bounds) {
        var n = bounds.getNorth().toFixed(4),
            s = bounds.getSouth().toFixed(4);
        var e = bounds.getEast().toFixed(4),
            w = bounds.getWest().toFixed(4);
        setText("roi-n", n + "° N");
        setText("roi-s", s + "° N");
        setText("roi-e", e + "° E");
        setText("roi-w", w + "° E");
        var midLat = (parseFloat(n) + parseFloat(s)) / 2;
        var km2 = (
            Math.abs(n - s) * 111 * Math.abs(e - w) * 111 * Math.cos((midLat * Math.PI) / 180)
        ).toFixed(0);
        setText("roi-area", Number(km2).toLocaleString() + " km²");
        document.getElementById("roi-placeholder").style.display = "none";
        document.getElementById("roi-details").style.display = "block";
    }

    function setText(id, val) {
        var el = document.getElementById(id);
        if (el) el.textContent = val;
    }
    function hide(id) {
        var el = document.getElementById(id);
        if (el) el.style.display = "none";
    }
    function show(id) {
        var el = document.getElementById(id);
        if (el) el.style.display = "block";
    }

    // ------------------------------------------------------------------
    // 5. Adaptive Satellite Product Search
    // ------------------------------------------------------------------
    document.getElementById("btn-search").addEventListener("click", function () {
        if (!currentBounds) return;

        var bbox = [
            currentBounds.getWest(),
            currentBounds.getSouth(),
            currentBounds.getEast(),
            currentBounds.getNorth(),
        ];

        show("results-panel");
        document.getElementById("product-list").innerHTML = "";
        hide("download-panel");
        footprintsGroup.clearLayers();
        selectedProduct = null;

        setSearchStatus("loading", "⏳ Searching Copernicus catalog for Sentinel-1 passes…");

        fetch("/api/satellite/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ bbox: bbox, max_days: 180 }),
        })
            .then(function (r) {
                return r.json();
            })
            .then(function (data) {
                if (data.error) {
                    setSearchStatus("error", "✗ " + data.error);
                    return;
                }
                handleSearchResult(data);
            })
            .catch(function (err) {
                setSearchStatus("error", "Network error: " + err.message);
            });
    });

    function handleSearchResult(data) {
        var products = data.products || [];
        var days = data.days_searched || 30;
        var expanded = data.expanded || false;
        var suggestions = data.suggestions || [];

        if (!products.length) {
            renderNoProducts(days, suggestions);
            return;
        }

        var msg = "✓ Found " + products.length + " Sentinel-1 pass(es)";
        if (expanded) {
            msg += " &nbsp;<span class='search-window-badge' style='background:rgba(55,224,196,0.15);padding:2px 6px;border-radius:4px;color:var(--signal);font-size:10px;'>searched " + days + " days</span>";
        }
        setSearchStatus("success", msg);
        renderProducts(products);
    }

    function setSearchStatus(type, html) {
        var el = document.getElementById("search-status");
        if (!el) return;
        var colors = {
            loading: "var(--ink-dim)",
            success: "var(--signal)",
            error: "var(--crimson)",
            warn: "var(--amber)",
        };
        el.innerHTML =
            '<div style="font-family:var(--font-mono);font-size:12px;padding:6px 0;color:' +
            (colors[type] || "var(--ink-dim)") +
            ';">' +
            html +
            "</div>";
    }

    // ------------------------------------------------------------------
    // 6. Product List & Swath Footprint Rendering
    // ------------------------------------------------------------------
    function renderProducts(products) {
        var listEl = document.getElementById("product-list");
        listEl.innerHTML = "";
        footprintsGroup.clearLayers();
        productLayers = {};

        products.forEach(function (p, i) {
            // Draw satellite footprint on the map if geometry is provided
            if (p.footprint) {
                try {
                    var layer = L.geoJSON(p.footprint, {
                        style: {
                            color: i === 0 ? "#37e0c4" : "#38bdf8",
                            weight: i === 0 ? 2.5 : 1.5,
                            fillColor: "#38bdf8",
                            fillOpacity: i === 0 ? 0.22 : 0.08,
                            dashArray: i === 0 ? null : "4",
                        },
                    });
                    layer.bindTooltip("<b>" + escHtml(p.name.substring(0, 26)) + "…</b><br>Mode: " + p.mode + " &middot; Orbit: " + p.orbit, {
                        className: "leaflet-popup-content",
                        sticky: true,
                    });
                    layer.on("click", function () {
                        selectProduct(p);
                        highlightProductCard(p.id);
                        highlightFootprint(p.id);
                    });
                    footprintsGroup.addLayer(layer);
                    productLayers[p.id] = layer;
                } catch (err) {
                    console.warn("Could not draw footprint for", p.name, err);
                }
            }

            // Create sidebar card
            var card = document.createElement("div");
            card.id = "card-" + p.id;
            card.className = "product-card" + (i === 0 ? " selected" : "");
            card.innerHTML =
                '<div class="product-name">' +
                escHtml(p.name) +
                "</div>" +
                '<div class="product-meta">' +
                "<span>" +
                (p.date || "—") +
                "</span>" +
                "<span>" +
                orbitLabel(p.orbit) +
                "</span>" +
                "<span>" +
                (p.mode || "IW") +
                "</span>" +
                "<span>" +
                (p.size_mb || "~474") +
                " MB</span>" +
                (p.footprint
                    ? '<span style="color:var(--signal);font-weight:700;">🗺 Mapped</span>'
                    : "") +
                "</div>";

            card.addEventListener("click", function () {
                highlightProductCard(p.id);
                selectProduct(p);
                highlightFootprint(p.id);
            });

            card.addEventListener("mouseenter", function () {
                highlightFootprint(p.id, false);
            });

            card.addEventListener("mouseleave", function () {
                if (!selectedProduct || selectedProduct.id !== p.id) {
                    resetFootprintStyle(p.id);
                }
            });

            listEl.appendChild(card);
        });

        // Auto-select first product
        if (products.length > 0) {
            selectProduct(products[0]);
            highlightFootprint(products[0].id);
        }
    }

    function highlightProductCard(id) {
        document.querySelectorAll(".product-card").forEach(function (c) {
            c.classList.remove("selected");
        });
        var target = document.getElementById("card-" + id);
        if (target) target.classList.add("selected");
    }

    function highlightFootprint(id, zoom) {
        Object.keys(productLayers).forEach(function (pid) {
            var l = productLayers[pid];
            if (pid === id) {
                l.setStyle({
                    color: "#37e0c4",
                    weight: 3,
                    fillColor: "#37e0c4",
                    fillOpacity: 0.28,
                    dashArray: null,
                });
                l.bringToFront();
                if (zoom) {
                    map.fitBounds(l.getBounds(), { padding: [40, 40] });
                }
            } else {
                l.setStyle({
                    color: "#38bdf8",
                    weight: 1.2,
                    fillColor: "#38bdf8",
                    fillOpacity: 0.06,
                    dashArray: "4",
                });
            }
        });
    }

    function resetFootprintStyle(id) {
        var l = productLayers[id];
        if (l) {
            l.setStyle({
                color: "#38bdf8",
                weight: 1.5,
                fillColor: "#38bdf8",
                fillOpacity: 0.08,
                dashArray: "4",
            });
        }
    }

    function renderNoProducts(daysSearched, suggestions) {
        setSearchStatus(
            "warn",
            "⚠ No passes found in this area in the past <b>" + daysSearched + " days</b>"
        );

        var html =
            '<div class="suggestions-box" style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:6px;padding:12px;margin-top:10px;">' +
            '<div class="suggestions-title" style="font-family:var(--font-mono);font-size:11px;font-weight:700;color:var(--amber);margin-bottom:6px;">Guidance</div>' +
            '<ul class="suggestions-list" style="font-family:var(--font-mono);font-size:11px;color:var(--ink-dim);padding-left:16px;line-height:1.6;margin:0;">';
        suggestions.forEach(function (s) {
            html += "<li>" + escHtml(s) + "</li>";
        });
        html += "</ul></div>";
        document.getElementById("product-list").innerHTML = html;
        hide("download-panel");
    }

    function orbitLabel(orbit) {
        if (!orbit || orbit === "—") return "—";
        return orbit === "ASCENDING"
            ? '<span style="color:var(--signal)">↑ ASC</span>'
            : '<span style="color:var(--amber)">↓ DESC</span>';
    }

    function escHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    // ------------------------------------------------------------------
    // 7. Product Selection & Dual Ingestion Actions
    // ------------------------------------------------------------------
    function selectProduct(p) {
        selectedProduct = p;
        show("download-panel");
        document.getElementById("selected-info").innerHTML =
            '<div style="font-family:var(--font-mono);font-size:11px;color:var(--ink-dim);margin-bottom:12px;">' +
            '<div style="color:var(--ink);font-weight:700;margin-bottom:4px;word-break:break-all;">' +
            escHtml(p.name) +
            "</div>" +
            "<div>Date: " +
            (p.date || "—") +
            "&nbsp;&middot;&nbsp;Orbit: " +
            (p.orbit || "—") +
            "&nbsp;&middot;&nbsp;Mode: " +
            (p.mode || "IW") +
            "&nbsp;&middot;&nbsp;~" +
            (p.size_mb || "474") +
            " MB</div>" +
            "</div>";

        document.getElementById("dl-result").innerHTML = "";
        hide("dl-steps");
        document.getElementById("btn-download").style.display = "block";
        document.getElementById("btn-download").textContent = "🌐 Download Full Raw Scene (Background)";
        resetSteps();
    }

    // --- Action A: ⚡ Instant 3-Second Demo Detection ---
    var btnDemo = document.getElementById("btn-demo-analyze");
    if (btnDemo) {
        btnDemo.addEventListener("click", function () {
            var demoStatus = document.getElementById("demo-status");
            btnDemo.disabled = true;
            btnDemo.textContent = "⏳ Running AI Pipeline…";
            if (demoStatus) {
                demoStatus.innerHTML = '<span style="color:var(--signal)">Processing Sentinel-1 SAR scene &middot; 3s…</span>';
            }

            fetch("/api/satellite/demo-scene", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ scene_type: "spill" }),
            })
                .then(function (r) {
                    return r.json();
                })
                .then(function (data) {
                    if (data.status === "success" && data.redirect_url) {
                        if (demoStatus) {
                            demoStatus.innerHTML = '<span style="color:var(--signal)">✓ Detection complete! Opening report…</span>';
                        }
                        setTimeout(function () {
                            window.location.href = data.redirect_url;
                        }, 400);
                    } else {
                        btnDemo.disabled = false;
                        btnDemo.textContent = "⚡ Run Instant Detection (3s Demo)";
                        if (demoStatus) {
                            demoStatus.innerHTML = '<span style="color:var(--crimson)">✗ Error: ' + (data.error || "Execution failed") + '</span>';
                        }
                    }
                })
                .catch(function (err) {
                    btnDemo.disabled = false;
                    btnDemo.textContent = "⚡ Run Instant Detection (3s Demo)";
                    if (demoStatus) {
                        demoStatus.innerHTML = '<span style="color:var(--crimson)">✗ Network error: ' + err.message + '</span>';
                    }
                });
        });
    }

    // --- Action B: 🌐 Full Background Raw Downloader ---
    var _pollTimer = null;

    document.getElementById("btn-download").addEventListener("click", function () {
        if (!selectedProduct) return;

        if (_pollTimer) {
            clearInterval(_pollTimer);
            _pollTimer = null;
        }

        this.style.display = "none";
        show("dl-steps");
        resetSteps();
        setStep("s-auth", "running");

        var sizeMb = (selectedProduct.size_mb || "~474").toString().replace("~", "");
        var dlSpan = document.getElementById("s-download-text") || document.querySelector("#s-download span:last-child");
        if (dlSpan) dlSpan.textContent = "Downloading satellite data (~" + sizeMb + " MB)";

        document.getElementById("dl-result").innerHTML =
            '<div style="font-family:var(--font-mono);font-size:11px;color:var(--ink-dim);' +
            'background:rgba(55,224,196,0.05);border:1px solid var(--line);' +
            'border-radius:6px;padding:10px 12px;margin-bottom:8px;">' +
            '&#9432; Running in background — you can continue using the app.<br>' +
            "A full Sentinel-1 scene (~" +
            sizeMb +
            " MB) takes 10–25 min." +
            "</div>";

        var url = "/satellite/acquire?product_id=" + encodeURIComponent(selectedProduct.id);

        fetch(url, { method: "POST", headers: { "Content-Type": "application/json" } })
            .then(function (r) {
                return r.json();
            })
            .then(function (data) {
                if (!data.job_id) {
                    showDownloadError(data.message || "Failed to start job.");
                    document.getElementById("btn-download").style.display = "block";
                    return;
                }
                _pollTimer = setInterval(function () {
                    pollJob(data.job_id);
                }, 4000);
            })
            .catch(function (err) {
                showDownloadError("Network error: " + escHtml(err.message));
                document.getElementById("btn-download").style.display = "block";
            });
    });

    var _stepMap = {
        authenticating: "s-auth",
        searching: "s-search",
        downloading: "s-download",
        extracting: "s-extract",
        preprocessing: "s-preprocess",
        complete: "s-preprocess",
        failed: null,
    };

    function pollJob(jobId) {
        fetch("/api/satellite/job/" + jobId)
            .then(function (r) {
                return r.json();
            })
            .then(function (job) {
                updateStepsFromJob(job);

                if (job.status === "complete") {
                    clearInterval(_pollTimer);
                    _pollTimer = null;
                    var res = job.result || {};
                    document.getElementById("dl-result").innerHTML =
                        '<div style="background:rgba(55,224,196,0.08);border:1px solid rgba(55,224,196,0.3);' +
                        'border-radius:7px;padding:10px 12px;font-family:var(--font-mono);font-size:11px;">' +
                        '<div style="color:var(--signal);font-weight:700;margin-bottom:4px;">✓ Download complete</div>' +
                        '<div style="color:var(--ink-dim);">' +
                        escHtml(res.preprocessing_note || "") +
                        "</div>" +
                        '<div style="margin-top:8px;"><a href="/sentinel1/scan" style="color:var(--signal);font-weight:700;">→ Run Full Pipeline Analysis</a></div>' +
                        "</div>";
                    document.getElementById("btn-download").style.display = "block";
                    document.getElementById("btn-download").textContent = "🌐 Download Full Raw Scene";
                }

                if (job.status === "failed") {
                    clearInterval(_pollTimer);
                    _pollTimer = null;
                    showDownloadError(job.error || job.message || "Acquisition failed.");
                    document.getElementById("btn-download").style.display = "block";
                    document.getElementById("btn-download").textContent = "🌐 Download Full Raw Scene";
                }
            })
            .catch(function () {});
    }

    function updateStepsFromJob(job) {
        var status = job.status;
        var order = ["authenticating", "searching", "downloading", "extracting", "preprocessing"];
        var currentIdx = order.indexOf(status);

        order.forEach(function (step, idx) {
            var elId = _stepMap[step];
            if (!elId) return;
            if (idx < currentIdx) {
                setStep(elId, "done");
            } else if (idx === currentIdx) {
                setStep(elId, "running");
            }
        });

        // Live Download Progress Bar & Percentage display
        var progressWrap = document.getElementById("dl-progress-wrap");
        var progressFill = document.getElementById("dl-progress-fill");
        var pctLabel = document.getElementById("dl-pct-label");
        var bytesLabel = document.getElementById("dl-bytes-label");
        var dlText = document.getElementById("s-download-text");
        var fallbackSize = (selectedProduct && selectedProduct.size_mb) ? selectedProduct.size_mb.toString().replace("~", "") : "474";

        if (status === "downloading") {
            if (progressWrap) progressWrap.style.display = "block";
            var pct = job.percent != null ? job.percent : 0;
            var dlMb = job.downloaded_mb != null ? job.downloaded_mb : 0;
            var totMb = job.total_mb != null ? job.total_mb : fallbackSize;

            if (pctLabel) pctLabel.textContent = pct + "%";
            if (bytesLabel) bytesLabel.textContent = dlMb + " MB / ~" + totMb + " MB";
            if (progressFill) progressFill.style.width = Math.min(100, Math.max(3, pct)) + "%";
            if (dlText) dlText.textContent = "Downloading satellite data (~" + totMb + " MB) — " + pct + "%";
        } else if (currentIdx > order.indexOf("downloading") || status === "complete") {
            if (progressFill) progressFill.style.width = "100%";
            if (pctLabel) pctLabel.textContent = "100%";
            if (dlText) dlText.textContent = "Downloading satellite data (~" + fallbackSize + " MB) — 100%";
            if (progressWrap) {
                setTimeout(function () {
                    if (job.status !== "downloading") progressWrap.style.display = "none";
                }, 1500);
            }
        } else {
            if (progressWrap) progressWrap.style.display = "none";
        }

        if (status === "complete") {
            ["s-auth", "s-search", "s-download", "s-extract", "s-preprocess"].forEach(function (id) {
                setStep(id, "done");
            });
        }
        if (status === "failed") {
            var failedId = _stepMap[job.step] || null;
            if (failedId) setStep(failedId, "error");
        }
    }

    function showDownloadError(msg) {
        document.getElementById("dl-result").innerHTML =
            '<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);' +
            'border-radius:7px;padding:10px 12px;font-family:var(--font-mono);font-size:11px;color:var(--crimson);">' +
            "&#10007; " +
            msg +
            "</div>";
    }

    function resetSteps() {
        ["s-auth", "s-search", "s-download", "s-extract", "s-preprocess"].forEach(function (id) {
            setStep(id, "idle");
        });
        var progressWrap = document.getElementById("dl-progress-wrap");
        var progressFill = document.getElementById("dl-progress-fill");
        var pctLabel = document.getElementById("dl-pct-label");
        if (progressWrap) progressWrap.style.display = "none";
        if (progressFill) progressFill.style.width = "0%";
        if (pctLabel) pctLabel.textContent = "0%";
    }

    function setStep(id, state) {
        var el = document.getElementById(id);
        if (!el) return;
        el.className = "dl-step " + state;
        var icon = el.querySelector(".dl-icon");
        if (!icon) return;
        if (state === "running") {
            icon.innerHTML = '<span class="dl-spinner"></span>';
        } else if (state === "done") {
            icon.textContent = "✓";
        } else if (state === "error") {
            icon.textContent = "✗";
        } else if (state === "warn") {
            icon.textContent = "⚠";
        } else {
            icon.textContent = "○";
        }
    }

    // ------------------------------------------------------------------
    // 8. Historical Oil Spill Incidents Layer
    // ------------------------------------------------------------------
    function loadIncidentHistory() {
        fetch("/api/history?limit=50")
            .then(function (r) {
                return r.json();
            })
            .then(function (records) {
                incidentsGroup.clearLayers();
                var countEl = document.getElementById("incident-count");
                if (countEl) countEl.textContent = records.length;

                records.forEach(function (rec) {
                    var lat = rec.lat || 18.9;
                    var lon = rec.lon || 72.3;
                    var status = rec.status || "";
                    var sev = (rec.severity || "LOW").toUpperCase();
                    var isSpill = status.indexOf("DETECTED") !== -1;

                    var color = "#22c55e"; // clear
                    var badgeClass = "badge-CLEAR";
                    if (isSpill) {
                        if (sev === "HIGH") {
                            color = "#ef4444";
                            badgeClass = "badge-HIGH";
                        } else if (sev === "MEDIUM") {
                            color = "#f59e0b";
                            badgeClass = "badge-MEDIUM";
                        } else {
                            color = "#37e0c4";
                            badgeClass = "badge-LOW";
                        }
                    }

                    var marker = L.circleMarker([lat, lon], {
                        radius: isSpill ? 8 : 5,
                        color: "#ffffff",
                        weight: 1.5,
                        fillColor: color,
                        fillOpacity: 0.9,
                    });

                    var thumbHtml = "";
                    if (rec.mask_image) {
                        thumbHtml =
                            '<img class="incident-thumb" src="/static/uploads/' +
                            encodeURIComponent(rec.mask_image) +
                            '" alt="Mask">';
                    } else if (rec.original_image) {
                        thumbHtml =
                            '<img class="incident-thumb" src="/static/uploads/' +
                            encodeURIComponent(rec.original_image) +
                            '" alt="Scene">';
                    }

                    var popupHtml =
                        '<div class="incident-popup">' +
                        '<h4><span style="color:' +
                        color +
                        ';">●</span> ' +
                        escHtml(rec.location_name || rec.record_id) +
                        "</h4>" +
                        '<div class="incident-badge ' +
                        badgeClass +
                        '">' +
                        escHtml(rec.status) +
                        "</div>" +
                        '<div style="font-size:10px;color:var(--ink-dim);margin-bottom:4px;">⏱ ' +
                        escHtml(rec.timestamp || "") +
                        "</div>" +
                        '<div style="font-size:11px;margin-bottom:4px;">' +
                        "<b>Area:</b> " +
                        (rec.area_km2 || 0) +
                        " km² (" +
                        (rec.area_pct || 0) +
                        "%)<br>" +
                        "<b>Confidence:</b> " +
                        (rec.confidence ? (rec.confidence * 100).toFixed(1) + "%" : "—") +
                        "</div>" +
                        thumbHtml +
                        '<a class="incident-link" href="/results/' +
                        encodeURIComponent(rec.record_id) +
                        '">→ View Full Detection Report</a>' +
                        "</div>";

                    marker.bindPopup(popupHtml, { maxWidth: 260 });
                    incidentsGroup.addLayer(marker);
                });
            })
            .catch(function (err) {
                console.warn("Could not load incident history:", err);
            });
    }

    // Toggle incidents layer
    var toggleIncidents = document.getElementById("toggle-incidents");
    if (toggleIncidents) {
        toggleIncidents.addEventListener("change", function () {
            if (this.checked) {
                map.addLayer(incidentsGroup);
            } else {
                map.removeLayer(incidentsGroup);
            }
        });
    }

    // Load incidents when map initializes
    loadIncidentHistory();

    // Trigger initial preset (Arabian Sea - Mumbai High) for smooth first impression
    var firstPreset = document.querySelector(".preset-btn");
    if (firstPreset) {
        firstPreset.click();
    }
})();

