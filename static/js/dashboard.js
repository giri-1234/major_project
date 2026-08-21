/**
 * dashboard.js
 * Small progressive-enhancement script for the upload console:
 *  - shows the selected filename
 *  - supports drag-and-drop onto the dropzone
 *  - shows an indeterminate-to-complete progress bar while the form
 *    submits (the actual processing happens server-side; this just gives
 *    the user feedback that analysis is running).
 */

(function () {
    const fileInput = document.getElementById("file-field");
    const fileChip = document.getElementById("file-name-display");
    const dropzone = document.getElementById("dropzone");
    const form = document.getElementById("upload-form");
    const progressWrap = document.getElementById("progress-wrap");
    const progressBar = document.getElementById("progress-bar");
    const progressPct = document.getElementById("progress-pct");
    const submitBtn = document.getElementById("submit-btn");

    if (!fileInput) return;

    function showFileName(file) {
        if (file) {
            fileChip.textContent = "Selected: " + file.name;
        }
    }

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            showFileName(fileInput.files[0]);
        }
    });

    if (dropzone) {
        ["dragenter", "dragover"].forEach((evt) =>
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add("dragover");
            })
        );
        ["dragleave", "drop"].forEach((evt) =>
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove("dragover");
            })
        );
        dropzone.addEventListener("drop", (e) => {
            const dropped = e.dataTransfer.files;
            if (dropped.length > 0) {
                fileInput.files = dropped;
                showFileName(dropped[0]);
            }
        });
    }

    if (form) {
        form.addEventListener("submit", () => {
            if (!fileInput.files || fileInput.files.length === 0) {
                return; // let the server render the "no file" error as before
            }
            progressWrap.classList.add("active");
            submitBtn.disabled = true;
            submitBtn.textContent = "Analyzing...";

            let pct = 0;
            const timer = setInterval(() => {
                // Ease toward 90%; the page will navigate away on real completion.
                pct += (90 - pct) * 0.15 + 1;
                if (pct > 90) pct = 90;
                progressBar.style.width = pct + "%";
                progressPct.textContent = Math.round(pct) + "%";
            }, 200);

            window.addEventListener("beforeunload", () => clearInterval(timer));
        });
    }
})();

// Satellite acquisition button
(function () {
    const acquireBtn = document.getElementById("acquire-btn");
    const acquireStatus = document.getElementById("acquire-status");

    if (!acquireBtn) return;

    acquireBtn.addEventListener("click", function () {
        acquireBtn.disabled = true;
        acquireBtn.textContent = "⏳ Acquiring satellite data...";
        acquireStatus.innerHTML = '<span style="color: var(--ink-dim); font-family: var(--font-mono); font-size: 12px;">Connecting to Copernicus Data Space Ecosystem...</span>';

        fetch("/satellite/acquire", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        })
        .then(function (res) { return res.json().then(function(data) { return { ok: res.ok, status: res.status, data: data }; }); })
        .then(function (result) {
            if (result.ok) {
                var d = result.data;
                var prepColor = d.preprocessing_status === "complete" ? "var(--signal)"
                              : d.preprocessing_status === "failed"   ? "var(--amber)"
                              : "var(--ink-dim)";
                var prepIcon  = d.preprocessing_status === "complete" ? "✓"
                              : d.preprocessing_status === "failed"   ? "⚠"
                              : "–";
                acquireStatus.innerHTML =
                    '<div style="background: rgba(55,224,196,0.08); border: 1px solid rgba(55,224,196,0.3); border-radius: 7px; padding: 14px 16px;">' +
                    '<div style="font-family: var(--font-mono); font-size: 12px; color: var(--signal); font-weight: 700; margin-bottom: 8px;">✓ Acquisition complete</div>' +
                    '<div style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-dim);">Product: ' + d.product_name + '</div>' +
                    '<div style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-dim); margin-top: 3px;">Raw TIFF: ' + d.raw_tiff_filename + '</div>' +
                    (d.processed_tiff_filename ? '<div style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-dim); margin-top: 3px;">Processed: ' + d.processed_tiff_filename + '</div>' : '') +
                    '<div style="font-family: var(--font-mono); font-size: 11px; margin-top: 6px; color: ' + prepColor + '">' + prepIcon + ' Preprocessing: ' + d.preprocessing_note + '</div>' +
                    '<div style="margin-top: 12px; font-size: 13px;">→ <a href="/sentinel1/scan" style="color: var(--signal);">Run Sentinel-1 scan to process this image</a></div>' +
                    '</div>';
                acquireBtn.textContent = "🛰 Check Latest Sentinel-1 Image";
            } else {
                acquireStatus.innerHTML =
                    '<div style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); border-radius: 7px; padding: 14px 16px;">' +
                    '<div style="font-family: var(--font-mono); font-size: 12px; color: var(--crimson); font-weight: 700; margin-bottom: 6px;">✗ ' + result.data.error_type + '</div>' +
                    '<div style="font-family: var(--font-mono); font-size: 11px; color: var(--ink-dim);">' + result.data.message + '</div>' +
                    '</div>';
                acquireBtn.textContent = "🛰 Check Latest Sentinel-1 Image";
            }
        })
        .catch(function (err) {
            acquireStatus.innerHTML =
                '<div style="font-family: var(--font-mono); font-size: 12px; color: var(--crimson);">Network error: ' + err.message + '</div>';
            acquireBtn.textContent = "🛰 Check Latest Sentinel-1 Image";
        })
        .finally(function () {
            acquireBtn.disabled = false;
        });
    });
})();
