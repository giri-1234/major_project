/**
 * map.js — Interactive Leaflet map for Sentinel-1 ROI selection
 *
 * Improvements over v1:
 *  - Adaptive search: automatically expands window 30→60→90→180 days
 *  - Shows all available products (up to 20), user picks which to download
 *  - Progress messaging during window expansion
 *  - Contextual suggestions when no products found after max window
 *  - Download targets the user-selected product by ID
 */
(function () {
    "use strict";

    // ------------------------------------------------------------------
    // Map initialisation
    // ------------------------------------------------------------------
    var map = L.map("map", { center: [13.0, 73.0], zoom: 6 });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxZoom: 18,
    }).addTo(map);

    var drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    var drawControl = new L.Control.Draw({
        position: "topleft",
        draw: {
            rectangle: {
                shapeOptions: { color: "#37e0c4", weight: 2, fillColor: "#37e0c4", fillOpacity: 0.15 },
                showArea: true,
            },
            polygon: false, polyline: false, circle: false,
            circlemarker: false, marker: false,
        },
        edit: { featureGroup: drawnItems, remove: true },
    });
    map.addControl(drawControl);

    var currentBounds = null;
    var selectedProduct = null;

    // ------------------------------------------------------------------
    // Draw events
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
        hide("results-panel"); hide("download-panel");
    });

    map.on(L.Draw.Event.EDITED, function () {
        drawnItems.eachLayer(function (l) { currentBounds = l.getBounds(); showROI(currentBounds); });
    });

    // ------------------------------------------------------------------
    // Toolbar buttons
    // ------------------------------------------------------------------
    document.getElementById("btn-draw").addEventListener("click", function () {
        var h = new L.Draw.Rectangle(map, drawControl.options.draw.rectangle);
        h.enable();
        this.classList.add("active");
        this.textContent = "✏ Click & drag on map…";
    });

    document.getElementById("btn-clear").addEventListener("click", function () {
        drawnItems.clearLayers();
        currentBounds = null;
        document.getElementById("roi-placeholder").style.display = "block";
        document.getElementById("roi-details").style.display = "none";
        hide("results-panel"); hide("download-panel");
    });

    document.getElementById("btn-use-view").addEventListener("click", function () {
        drawnItems.clearLayers();
        currentBounds = map.getBounds();
        L.rectangle(currentBounds, {
            color: "#37e0c4", weight: 2,
            fillColor: "#37e0c4", fillOpacity: 0.15,
        }).addTo(drawnItems);
        showROI(currentBounds);
    });

    // Preset buttons
    document.querySelectorAll(".preset-btn").forEach(function (b) {
        b.addEventListener("click", function () {
            map.setView(
                [parseFloat(b.dataset.lat), parseFloat(b.dataset.lng)],
                parseInt(b.dataset.zoom)
            );
        });
    });

    // ------------------------------------------------------------------
    // ROI display helpers
    // ------------------------------------------------------------------
    function showROI(bounds) {
        var n = bounds.getNorth().toFixed(4), s = bounds.getSouth().toFixed(4);
        var e = bounds.getEast().toFixed(4),  w = bounds.getWest().toFixed(4);
        setText("roi-n", n + "° N"); setText("roi-s", s + "° N");
        setText("roi-e", e + "° E"); setText("roi-w", w + "° E");
        var midLat = (parseFloat(n) + parseFloat(s)) / 2;
        var km2 = (
            Math.abs(n - s) * 111 *
            Math.abs(e - w) * 111 * Math.cos(midLat * Math.PI / 180)
        ).toFixed(0);
        setText("roi-area", Number(km2).toLocaleString() + " km²");
        document.getElementById("roi-placeholder").style.display = "none";
        document.getElementById("roi-details").style.display = "block";
    }

    function setText(id, val) {
        var el = document.getElementById(id);
        if (el) el.textContent = val;
    }
    function hide(id) { var el = document.getElementById(id); if (el) el.style.display = "none"; }
    function show(id) { var el = document.getElementById(id); if (el) el.style.display = "block"; }

    // ------------------------------------------------------------------
    // Adaptive search
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
        selectedProduct = null;

        setSearchStatus("loading", "⏳ Searching last 30 days…");

        fetch("/api/satellite/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ bbox: bbox, max_days: 180 }),
        })
        .then(function (r) { return r.json(); })
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
        var products    = data.products    || [];
        var days        = data.days_searched || 30;
        var expanded    = data.expanded    || false;
        var suggestions = data.suggestions || [];

        if (!products.length) {
            renderNoProducts(days, suggestions);
            return;
        }

        var msg = "✓ Found " + products.length + " product(s)";
        if (expanded) {
            msg += " &nbsp;<span class='search-window-badge'>searched " + days + " days</span>";
        }
        setSearchStatus("success", msg);
        renderProducts(products);
    }

    function setSearchStatus(type, html) {
        var el = document.getElementById("search-status");
        if (!el) return;
        var colors = { loading: "var(--ink-dim)", success: "var(--signal)", error: "var(--crimson)", warn: "var(--amber)" };
        el.innerHTML = '<div style="font-family:var(--font-mono);font-size:12px;padding:6px 0;color:' +
            (colors[type] || "var(--ink-dim)") + ';">' + html + '</div>';
    }

    // ------------------------------------------------------------------
    // Product list rendering
    // ------------------------------------------------------------------
    function renderProducts(products) {
        var listEl = document.getElementById("product-list");
        listEl.innerHTML = "";

        products.forEach(function (p, i) {
            var card = document.createElement("div");
            card.className = "product-card" + (i === 0 ? " selected" : "");
            card.innerHTML =
                '<div class="product-name">' + escHtml(p.name) + '</div>' +
                '<div class="product-meta">' +
                    '<span>' + (p.date || "—") + '</span>' +
                    '<span>' + orbitLabel(p.orbit) + '</span>' +
                    '<span>' + (p.mode || "IW") + '</span>' +
                    '<span>' + (p.size_mb || "~474") + ' MB</span>' +
                '</div>';
            card.addEventListener("click", function () {
                document.querySelectorAll(".product-card").forEach(function (c) {
                    c.classList.remove("selected");
                });
                card.classList.add("selected");
                selectProduct(p);
            });
            listEl.appendChild(card);
        });

        // Auto-select first
        selectProduct(products[0]);
    }

    function renderNoProducts(daysSearched, suggestions) {
        setSearchStatus("warn",
            "⚠ No products found after searching <b>" + daysSearched + " days</b>"
        );

        var html = '<div class="suggestions-box">' +
            '<div class="suggestions-title">Suggestions</div>' +
            '<ul class="suggestions-list">';
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
    // Product selection & download
    // ------------------------------------------------------------------
    function selectProduct(p) {
        selectedProduct = p;
        show("download-panel");
        document.getElementById("selected-info").innerHTML =
            '<div style="font-family:var(--font-mono);font-size:11px;color:var(--ink-dim);margin-bottom:10px;">' +
            '<div style="color:var(--ink);font-weight:700;margin-bottom:5px;word-break:break-all;">' + escHtml(p.name) + '</div>' +
            '<div>Date: ' + (p.date || "—") +
            '&nbsp;&nbsp;Orbit: ' + (p.orbit || "—") +
            '&nbsp;&nbsp;Mode: ' + (p.mode || "IW") +
            '&nbsp;&nbsp;~' + (p.size_mb || "474") + ' MB</div>' +
            '</div>';
        document.getElementById("dl-result").innerHTML = "";
        hide("dl-steps");
        document.getElementById("btn-download").style.display = "block";
        document.getElementById("btn-download").textContent = "🛰 Download & Analyze";
        resetSteps();
    }

    // ------------------------------------------------------------------
    // Download with background job + polling
    // ------------------------------------------------------------------
    var _pollTimer = null;

    document.getElementById("btn-download").addEventListener("click", function () {
        if (!selectedProduct) return;

        // Clear any previous poll timer
        if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }

        this.style.display = "none";
        show("dl-steps");
        resetSteps();
        setStep("s-auth", "running");

        // Update download step label with actual size
        var sizeMb = (selectedProduct.size_mb || "~474").toString().replace("~", "");
        var dlSpan = document.querySelector("#s-download span:last-child");
        if (dlSpan) dlSpan.textContent = "Downloading satellite data (~" + sizeMb + " MB)";

        document.getElementById("dl-result").innerHTML =
            '<div style="font-family:var(--font-mono);font-size:11px;color:var(--ink-dim);' +
            'background:rgba(55,224,196,0.05);border:1px solid var(--line);' +
            'border-radius:6px;padding:10px 12px;margin-bottom:8px;">' +
            '&#9432; Running in background — you can keep using the page.<br>' +
            'A full Sentinel-1 scene (~' + sizeMb + ' MB) takes 10–30 min.' +
            '</div>';

        var url = "/satellite/acquire?product_id=" + encodeURIComponent(selectedProduct.id);

        // Step 1: POST to start the job — returns immediately with job_id
        fetch(url, { method: "POST", headers: { "Content-Type": "application/json" } })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (!data.job_id) {
                showDownloadError(data.message || "Failed to start job.");
                document.getElementById("btn-download").style.display = "block";
                return;
            }
            // Step 2: poll the job status every 4 seconds
            _pollTimer = setInterval(function () {
                pollJob(data.job_id);
            }, 4000);
        })
        .catch(function (err) {
            showDownloadError("Network error: " + escHtml(err.message));
            document.getElementById("btn-download").style.display = "block";
        });
    });

    // Map server-side step names to UI step IDs
    var _stepMap = {
        "authenticating": "s-auth",
        "searching":      "s-search",
        "downloading":    "s-download",
        "extracting":     "s-extract",
        "preprocessing":  "s-preprocess",
        "complete":       "s-preprocess",
        "failed":         null,
    };

    function pollJob(jobId) {
        fetch("/api/satellite/job/" + jobId)
        .then(function (r) { return r.json(); })
        .then(function (job) {
            updateStepsFromJob(job);

            if (job.status === "complete") {
                clearInterval(_pollTimer); _pollTimer = null;
                var res = job.result || {};
                document.getElementById("dl-result").innerHTML =
                    '<div style="background:rgba(55,224,196,0.08);border:1px solid rgba(55,224,196,0.3);' +
                    'border-radius:7px;padding:10px 12px;font-family:var(--font-mono);font-size:11px;">' +
                    '<div style="color:var(--signal);font-weight:700;margin-bottom:4px;">✓ Download complete</div>' +
                    '<div style="color:var(--ink-dim);">' + escHtml(res.preprocessing_note || "") + '</div>' +
                    '<div style="margin-top:8px;"><a href="/sentinel1/scan" style="color:var(--signal);">→ Run Analysis</a></div>' +
                    '</div>';
                document.getElementById("btn-download").style.display = "block";
                document.getElementById("btn-download").textContent = "🛰 Download & Analyze";
            }

            if (job.status === "failed") {
                clearInterval(_pollTimer); _pollTimer = null;
                showDownloadError(job.error || job.message || "Acquisition failed.");
                document.getElementById("btn-download").style.display = "block";
                document.getElementById("btn-download").textContent = "🛰 Download & Analyze";
            }
        })
        .catch(function () {
            // Network blip — keep polling, don't abort
        });
    }

    function updateStepsFromJob(job) {
        var status = job.status;
        // Mark all steps before the current one as done
        var order = ["authenticating","searching","downloading","extracting","preprocessing"];
        var currentIdx = order.indexOf(status);

        order.forEach(function (step, idx) {
            var elId = _stepMap[step];
            if (!elId) return;
            if (idx < currentIdx) {
                setStep(elId, "done");
            } else if (idx === currentIdx) {
                setStep(elId, "running");
            }
            // future steps stay idle
        });

        if (status === "complete") {
            ["s-auth","s-search","s-download","s-extract","s-preprocess"]
                .forEach(function (id) { setStep(id, "done"); });
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
            '&#10007; ' + msg + '</div>';
    }

    // ------------------------------------------------------------------
    // Download step indicator helpers
    // ------------------------------------------------------------------
    function resetSteps() {
        ["s-auth", "s-search", "s-download", "s-extract", "s-preprocess"]
            .forEach(function (id) { setStep(id, "idle"); });
    }

    function setStep(id, state) {
        var el = document.getElementById(id);
        if (!el) return;
        el.className = "dl-step " + state;
        var icon = el.querySelector(".dl-icon");
        var icons = { idle: "○", running: "⏳", done: "✓", warn: "⚠", error: "✗" };
        if (icon) icon.textContent = icons[state] || "○";
    }

})();
