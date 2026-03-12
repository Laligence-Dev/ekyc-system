// ===== State =====
const state = {
    currentStep: 1,
    sessionId: null,
    selectedFile: null,
    stream: null,
    mrz: null,
};

// ===== DOM Helpers =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dropZone         = $("#drop-zone");
const fileInput        = $("#file-input");
const previewContainer = $("#preview-container");
const docPreview       = $("#doc-preview");
const btnUpload        = $("#btn-upload");
const btnText          = btnUpload.querySelector(".btn-text");
const btnLoader        = btnUpload.querySelector(".btn-loader");
const uploadError      = $("#upload-error");

const webcam        = $("#webcam");
const captureCanvas = $("#capture-canvas");
const statusDot     = $("#status-dot");
const statusText    = $("#status-text");
const btnCapture    = $("#btn-capture");
const cameraError   = $("#camera-error");

// ===== Step Navigation =====
function goToStep(step) {
    state.currentStep = step;

    $$(".step").forEach((el) => {
        const s = parseInt(el.dataset.step);
        el.classList.toggle("active", s === step);
        el.classList.toggle("completed", s < step);
    });

    $$(".step-connector").forEach((c, i) => {
        c.classList.toggle("active", i + 1 < step);
    });

    const sections = ["step-upload", "step-camera", "step-result"];
    sections.forEach((id, i) => {
        const el = $(`#${id}`);
        el.classList.toggle("active",  i === step - 1);
        el.classList.toggle("hidden",  i !== step - 1);
    });

    if (step === 2) startCamera();
}

// ===== Step 1: Document Upload =====

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleFileSelected(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => { if (e.target.files[0]) handleFileSelected(e.target.files[0]); });

function handleFileSelected(file) {
    hideError(uploadError);
    if (!["image/jpeg", "image/png"].includes(file.type)) {
        showError(uploadError, "Please select a JPEG or PNG image.");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError(uploadError, "File is too large. Maximum size is 10MB.");
        return;
    }
    state.selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        docPreview.src = e.target.result;
        previewContainer.classList.remove("hidden");
        dropZone.classList.add("hidden");
        btnUpload.disabled = false;
    };
    reader.readAsDataURL(file);
}

$("#remove-doc").addEventListener("click", () => {
    state.selectedFile = null;
    fileInput.value = "";
    docPreview.src = "";
    previewContainer.classList.add("hidden");
    dropZone.classList.remove("hidden");
    btnUpload.disabled = true;
    hideError(uploadError);
});

btnUpload.addEventListener("click", async () => {
    if (!state.selectedFile) return;
    hideError(uploadError);
    btnUpload.disabled = true;
    btnText.textContent = "Processing...";
    btnLoader.classList.remove("hidden");

    const formData = new FormData();
    formData.append("document", state.selectedFile);

    try {
        const res  = await fetch("/api/document/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Upload failed.");
        state.sessionId = data.session_id;
        state.mrz = data.mrz || null;
        console.log("[MRZ]", JSON.stringify(state.mrz, null, 2));
        goToStep(2);
    } catch (err) {
        showError(uploadError, err.message);
        btnUpload.disabled = false;
    } finally {
        btnText.textContent = "Upload & Continue";
        btnLoader.classList.add("hidden");
    }
});

// ===== MRZ Display =====

function renderMrz(mrz) {
    console.log("[renderMrz]", mrz);

    if (!mrz || !mrz.found) {
        $("#mrz-not-found").classList.remove("hidden");
        $("#mrz-fields").innerHTML = "";
        $("#mrz-validity").textContent = "";
        return;
    }

    $("#mrz-not-found").classList.add("hidden");

    // Validity label
    const validityEl = $("#mrz-validity");
    if (mrz.valid) {
        validityEl.textContent = "Check digits valid ✓";
        validityEl.className = "mrz-validity valid";
    } else {
        validityEl.textContent = "Check digit mismatch";
        validityEl.className = "mrz-validity invalid";
    }

    // All possible fields — show even if empty so you can spot what's missing
    const fields = [
        { label: "Surname",          value: mrz.surname         },
        { label: "Given Names",      value: mrz.given_names     },
        { label: "Document Type",    value: mrz.document_type   },
        { label: "Country",          value: mrz.country         },
        { label: "Document Number",  value: mrz.document_number },
        { label: "Nationality",      value: mrz.nationality     },
        { label: "Date of Birth",    value: mrz.date_of_birth   },
        { label: "Expiry Date",      value: mrz.expiry_date     },
        { label: "Sex",              value: mrz.sex             },
    ];

    $("#mrz-fields").innerHTML = fields.map(f => `
        <div class="mrz-field">
            <span class="mrz-field-label">${f.label}</span>
            <span class="mrz-field-value">${f.value || '<span style="color:#475569">—</span>'}</span>
        </div>
    `).join("");
}

// ===== Step 2: Camera + Take Selfie =====

async function startCamera() {
    updateStatus("Initializing camera...", "scanning");
    btnCapture.disabled = true;
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        });
        webcam.srcObject = state.stream;
        await webcam.play();
        updateStatus("Ready — click the button to take a selfie", "detected");
        btnCapture.disabled = false;
    } catch (err) {
        updateStatus("Camera access denied. Please allow camera permission.", "error");
    }
}

btnCapture.addEventListener("click", async () => {
    hideError(cameraError);
    btnCapture.disabled = true;
    $(".btn-capture-text").textContent = "Verifying...";
    $(".capture-circle").classList.add("hidden");
    updateStatus("Capturing & verifying...", "scanning");

    try {
        const canvas = captureCanvas;
        const ctx = canvas.getContext("2d");
        canvas.width  = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        if (canvas.width === 0) throw new Error("Camera not ready. Please wait and try again.");
        ctx.drawImage(webcam, 0, 0);

        const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.8));
        if (!blob) throw new Error("Failed to capture image.");

        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        formData.append("session_id", state.sessionId);

        const res  = await fetch("/api/verify/frame", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            if (res.status === 404) throw new Error("Session expired. Please start over.");
            throw new Error(data.detail || "Verification failed.");
        }

        if (!data.face_detected) {
            showError(cameraError, "No face detected. Make sure your face is clearly visible and try again.");
            updateStatus("Ready — click the button to take a selfie", "detected");
            btnCapture.disabled = false;
        } else if (data.is_live === false) {
            showError(cameraError, "Spoof detected! Please show your real face, not a photo or screen.");
            updateStatus("Liveness check failed — try again with your real face", "error");
            btnCapture.disabled = false;
        } else if (data.match) {
            updateStatus("Match found!", "matched");
            stopCamera();
            showSuccessResult(data.confidence);
        } else {
            const pct = Math.round(data.confidence * 100);
            stopCamera();
            showFailureResult(`Face does not match the document (${pct}% similarity). Please try again.`);
        }
    } catch (err) {
        showError(cameraError, err.message);
        updateStatus("Ready — click the button to take a selfie", "detected");
        btnCapture.disabled = false;
    } finally {
        $(".btn-capture-text").textContent = "Take Selfie";
        $(".capture-circle").classList.remove("hidden");
    }
});

function updateStatus(text, type) {
    statusText.textContent = text;
    statusDot.className = `status-dot ${type}`;
}

function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach((t) => t.stop());
        state.stream = null;
    }
}

// ===== Step 3: Results =====

function showSuccessResult(confidence) {
    const pct = Math.round(confidence * 100);
    goToStep(3);
    $("#result-success").classList.remove("hidden");
    $("#result-failure").classList.add("hidden");
    $("#confidence-score").textContent = `${pct}%`;
    renderMrz(state.mrz);
}

function showFailureResult(message) {
    goToStep(3);
    $("#result-success").classList.add("hidden");
    $("#result-failure").classList.remove("hidden");
    $("#failure-message").textContent = message;
    renderMrz(state.mrz);
}

$("#btn-retry").addEventListener("click", () => location.reload());

// ===== Shared Helpers =====

function showError(el, msg) {
    el.textContent = msg;
    el.classList.remove("hidden");
}
function hideError(el) {
    el.classList.add("hidden");
}
