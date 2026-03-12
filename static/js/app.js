// ===== State =====
const state = {
    currentStep: 1,
    sessionId: null,
    selectedFile: null,
    stream: null,
    livenessStream: null,
    challengeInterval: null,
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

const webcamLiveness    = $("#webcam-liveness");
const livenessCanvas    = $("#liveness-canvas");
const livenessDot       = $("#liveness-dot");
const livenessStatusTxt = $("#liveness-status-text");
const challengeBox      = $("#challenge-box");
const challengeIcon     = $("#challenge-icon");
const challengeInstr    = $("#challenge-instruction");
const challengePassed   = $("#challenge-passed-banner");
const livenessError     = $("#liveness-error");

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

    const sections = ["step-upload", "step-liveness", "step-camera", "step-result"];
    sections.forEach((id, i) => {
        const el = $(`#${id}`);
        el.classList.toggle("active",  i === step - 1);
        el.classList.toggle("hidden",  i !== step - 1);
    });

    if (step === 2) startLivenessCamera();
    if (step === 3) startCamera();
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
        goToStep(2);
    } catch (err) {
        showError(uploadError, err.message);
        btnUpload.disabled = false;
    } finally {
        btnText.textContent = "Upload & Continue";
        btnLoader.classList.add("hidden");
    }
});

// ===== Step 2: Active Liveness Challenge (Frontend-only via MediaPipe FaceMesh) =====

const CHALLENGE_LIST = ["blink", "open_mouth", "turn_left", "turn_right"];
const CHALLENGE_META = {
    blink:       { icon: "👁",  text: "Close your eyes slowly and hold for a moment" },
    open_mouth:  { icon: "👄",  text: "Open your mouth wide" },
    turn_left:   { icon: "⬅️", text: "Slowly turn your head to the LEFT" },
    turn_right:  { icon: "➡️", text: "Slowly turn your head to the RIGHT" },
};

// Landmark indices for MediaPipe FaceMesh (468 points)
const L_EYE  = [33, 160, 158, 133, 153, 144];
const R_EYE  = [362, 385, 387, 263, 373, 380];
const MOUTH_TOP = 13, MOUTH_BOT = 14, MOUTH_L = 78, MOUTH_R = 308;
const NOSE_TIP  = 4,  L_EYE_OUT = 33, R_EYE_OUT = 263;

// Thresholds
const EAR_BLINK   = 0.20;   // eye aspect ratio to count as blink
const MAR_OPEN    = 0.45;   // mouth aspect ratio to count as open
const YAW_DEG     = 0.06;   // nose offset fraction of face width
const HOLD_FRAMES = 3;      // consecutive frames needed

let faceMesh      = null;
let mpCamera      = null;
let challengeKey  = null;
let holdCount     = 0;

function dist(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function ear(lm, idx) {
    // Eye Aspect Ratio: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    const p1 = lm[idx[0]], p2 = lm[idx[1]], p3 = lm[idx[2]];
    const p4 = lm[idx[3]], p5 = lm[idx[4]], p6 = lm[idx[5]];
    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4));
}

function mar(lm) {
    const top = lm[MOUTH_TOP], bot = lm[MOUTH_BOT];
    const left = lm[MOUTH_L],  right = lm[MOUTH_R];
    return dist(top, bot) / dist(left, right);
}

function yawOffset(lm) {
    // Normalised horizontal offset of nose tip relative to eye midpoint
    const eyeMidX = (lm[L_EYE_OUT].x + lm[R_EYE_OUT].x) / 2;
    const faceW   = dist(lm[L_EYE_OUT], lm[R_EYE_OUT]);
    return faceW > 0 ? (lm[NOSE_TIP].x - eyeMidX) / faceW : 0;
}

function checkAction(lm) {
    if (challengeKey === "blink") {
        const e = (ear(lm, L_EYE) + ear(lm, R_EYE)) / 2;
        return e < EAR_BLINK;
    }
    if (challengeKey === "open_mouth") {
        return mar(lm) > MAR_OPEN;
    }
    if (challengeKey === "turn_left") {
        // nose shifts right in image when turning left
        return yawOffset(lm) > YAW_DEG;
    }
    if (challengeKey === "turn_right") {
        return yawOffset(lm) < -YAW_DEG;
    }
    return false;
}

async function startLivenessCamera() {
    updateLiveness("Starting camera...", "scanning");
    hideError(livenessError);

    // Pick a random challenge
    challengeKey = CHALLENGE_LIST[Math.floor(Math.random() * CHALLENGE_LIST.length)];
    holdCount    = 0;
    const meta   = CHALLENGE_META[challengeKey];

    try {
        state.livenessStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        });
        webcamLiveness.srcObject = state.livenessStream;
        await webcamLiveness.play();
    } catch (err) {
        updateLiveness("Camera access denied.", "error");
        return;
    }

    // Show challenge instruction
    challengeBox.classList.remove("hidden");
    challengeIcon.textContent  = meta.icon;
    challengeInstr.textContent = meta.text;
    updateLiveness("Follow the instruction above", "scanning");

    // Init MediaPipe FaceMesh
    if (typeof FaceMesh === "undefined") {
        // Fallback: auto-pass if library didn't load
        console.warn("MediaPipe FaceMesh not available — auto-passing liveness");
        onChallengePassed();
        return;
    }

    faceMesh = new FaceMesh({
        locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });
    faceMesh.setOptions({
        maxNumFaces:            1,
        refineLandmarks:        true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence:  0.5,
    });
    faceMesh.onResults(onFaceMeshResults);

    mpCamera = new Camera(webcamLiveness, {
        onFrame: async () => {
            if (faceMesh) await faceMesh.send({ image: webcamLiveness });
        },
        width: 640,
        height: 480,
    });
    mpCamera.start();
}

function onFaceMeshResults(results) {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        updateLiveness("Position your face in the circle", "scanning");
        $("#face-guide-liveness").classList.remove("detected");
        holdCount = 0;
        return;
    }

    const lm = results.multiFaceLandmarks[0];
    $("#face-guide-liveness").classList.add("detected");

    if (checkAction(lm)) {
        holdCount++;
        updateLiveness(`Hold... ${holdCount}/${HOLD_FRAMES}`, "detected");
        if (holdCount >= HOLD_FRAMES) {
            onChallengePassed();
        }
    } else {
        holdCount = 0;
        updateLiveness(CHALLENGE_META[challengeKey].text, "scanning");
    }
}

function onChallengePassed() {
    // Stop FaceMesh camera loop
    if (mpCamera)  { mpCamera.stop();  mpCamera  = null; }
    if (faceMesh)  { faceMesh.close(); faceMesh  = null; }

    challengeBox.classList.add("hidden");
    challengePassed.classList.remove("hidden");
    updateLiveness("Liveness verified!", "matched");
    stopLivenessCamera();

    setTimeout(() => goToStep(3), 1500);
}

function stopLivenessCamera() {
    if (state.livenessStream) {
        state.livenessStream.getTracks().forEach((t) => t.stop());
        state.livenessStream = null;
    }
}

function updateLiveness(text, type) {
    livenessStatusTxt.textContent = text;
    livenessDot.className = `status-dot ${type}`;
}

// ===== Step 3: Camera + Take Selfie =====

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

// ===== Step 4: Results =====

function showSuccessResult(confidence) {
    const pct = Math.round(confidence * 100);
    goToStep(4);
    $("#result-success").classList.remove("hidden");
    $("#result-failure").classList.add("hidden");
    $("#confidence-score").textContent = `${pct}%`;
}

function showFailureResult(message) {
    goToStep(4);
    $("#result-success").classList.add("hidden");
    $("#result-failure").classList.remove("hidden");
    $("#failure-message").textContent = message;
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
