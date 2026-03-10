// ===== State =====
const state = {
    currentStep: 1,
    sessionId: null,
    selectedFile: null,
    stream: null,
};

// ===== DOM Helpers =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dropZone = $("#drop-zone");
const fileInput = $("#file-input");
const previewContainer = $("#preview-container");
const docPreview = $("#doc-preview");
const btnUpload = $("#btn-upload");
const btnText = btnUpload.querySelector(".btn-text");
const btnLoader = btnUpload.querySelector(".btn-loader");
const uploadError = $("#upload-error");

const webcam = $("#webcam");
const captureCanvas = $("#capture-canvas");
const statusDot = $("#status-dot");
const statusText = $("#status-text");
const btnCapture = $("#btn-capture");
const cameraError = $("#camera-error");

// ===== Step Navigation =====
function goToStep(step) {
    state.currentStep = step;

    $$(".step").forEach((el) => {
        const s = parseInt(el.dataset.step);
        el.classList.toggle("active", s === step);
        el.classList.toggle("completed", s < step);
    });

    const connectors = $$(".step-connector");
    connectors.forEach((c, i) => {
        c.classList.toggle("active", i + 1 < step);
    });

    const sections = ["step-upload", "step-camera", "step-result"];
    sections.forEach((id, i) => {
        const el = $(`#${id}`);
        el.classList.toggle("active", i === step - 1);
        el.classList.toggle("hidden", i !== step - 1);
    });

    if (step === 2) {
        startCamera();
    }
}

// ===== Step 1: Document Upload =====

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFileSelected(e.target.files[0]);
});

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
        const res = await fetch("/api/document/upload", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        state.sessionId = data.session_id;
        goToStep(2);
    } catch (err) {
        showError(uploadError, err.message);
        btnUpload.disabled = false;
    } finally {
        btnText.textContent = "Upload & Continue";
        btnLoader.classList.add("hidden");
    }
});

// ===== Step 2: Camera + Take Selfie =====

async function startCamera() {
    updateStatus("Initializing camera...", "scanning");
    btnCapture.disabled = true;

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
        });

        webcam.srcObject = state.stream;
        await webcam.play();

        updateStatus("Ready \u2014 click the button to take a selfie", "detected");
        btnCapture.disabled = false;
    } catch (err) {
        console.error("Camera error:", err);
        updateStatus("Camera access denied. Please allow camera permission.", "error");
    }
}

// Take Selfie button
btnCapture.addEventListener("click", async () => {
    hideError(cameraError);
    btnCapture.disabled = true;

    const captureText = $(".btn-capture-text");
    const captureCircle = $(".capture-circle");
    captureText.textContent = "Verifying...";
    captureCircle.classList.add("hidden");

    updateStatus("Capturing & verifying...", "scanning");

    try {
        const canvas = captureCanvas;
        const ctx = canvas.getContext("2d");

        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;

        if (canvas.width === 0 || canvas.height === 0) {
            throw new Error("Camera not ready. Please wait a moment and try again.");
        }

        ctx.drawImage(webcam, 0, 0);

        const blob = await new Promise((resolve) =>
            canvas.toBlob(resolve, "image/jpeg", 0.8)
        );

        if (!blob) {
            throw new Error("Failed to capture image.");
        }

        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        formData.append("session_id", state.sessionId);

        const res = await fetch("/api/verify/frame", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            if (res.status === 404) {
                throw new Error("Session expired. Please start over.");
            }
            throw new Error(data.detail || "Verification failed.");
        }

        if (!data.face_detected) {
            showError(cameraError, "No face detected. Make sure your face is clearly visible and try again.");
            updateStatus("Ready \u2014 click the button to take a selfie", "detected");
            btnCapture.disabled = false;
        } else if (data.is_live === false) {
            showError(cameraError, "Spoof detected! Please show your real face, not a photo or screen.");
            updateStatus("Liveness check failed \u2014 try again with your real face", "error");
            btnCapture.disabled = false;
        } else if (data.match) {
            // SUCCESS
            updateStatus("Match found!", "matched");
            stopCamera();
            showSuccessResult(data.confidence);
        } else {
            // No match
            const pct = Math.round(data.confidence * 100);
            stopCamera();
            showFailureResult(
                `Face does not match the document (${pct}% similarity). Please try again.`
            );
        }
    } catch (err) {
        showError(cameraError, err.message);
        updateStatus("Ready \u2014 click the button to take a selfie", "detected");
        btnCapture.disabled = false;
    } finally {
        captureText.textContent = "Take Selfie";
        captureCircle.classList.remove("hidden");
    }
});

function updateStatus(text, type) {
    statusText.textContent = text;
    statusDot.className = `status-dot ${type}`;
}

function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
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
}

function showFailureResult(message) {
    goToStep(3);
    $("#result-success").classList.add("hidden");
    $("#result-failure").classList.remove("hidden");
    $("#failure-message").textContent = message;
}

// Retry
$("#btn-retry").addEventListener("click", () => {
    location.reload();
});

// ===== Shared Helpers =====

function showError(el, msg) {
    el.textContent = msg;
    el.classList.remove("hidden");
}

function hideError(el) {
    el.classList.add("hidden");
}
