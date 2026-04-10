/**
 * SignBridge AI — app.js
 * Module 1: Gesture Detection (Patient Panel)
 *
 * - Captures webcam frames via Canvas
 * - Sends base64 JPEG to /api/detect every DETECT_INTERVAL ms
 * - Draws 21 hand landmarks onto overlay canvas
 * - Updates confidence bar, label, and chat history
 * - Triggers TTS via /api/speech
 */

(function () {
  "use strict";

  // ─── Config ────────────────────────────────────────────────────────────
  const DETECT_INTERVAL = 400;   // ms between API calls (<500ms latency target)
  const MIN_CONFIDENCE  = 0.50;  // threshold to register a detection
  const DEBOUNCE_SAME   = 2000;  // ms before repeating same sign in chat

  // ─── State ─────────────────────────────────────────────────────────────
  let _stream       = null;
  let _detectTimer  = null;
  let _lastLabel    = null;
  let _lastSpeakTs  = 0;
  let _isDetecting  = false;
  let _sessionId    = `sb_${Date.now()}`;

  // ─── DOM refs ──────────────────────────────────────────────────────────
  const video          = () => document.getElementById("webcam");
  const lmCanvas       = () => document.getElementById("landmark-canvas");
  const detectionLabel = () => document.getElementById("detection-label");
  const confBar        = () => document.getElementById("confidence-bar");
  const confText       = () => document.getElementById("confidence-text");
  const detectedText   = () => document.getElementById("detected-text");
  const btnStart       = () => document.getElementById("btn-start-camera");
  const btnStop        = () => document.getElementById("btn-stop-detect");
  const scanLine       = () => document.getElementById("scan-line");
  const dotDetect      = () => document.getElementById("dot-detect");
  const badgeDetect    = () => document.getElementById("badge-detect");

  // ─── Public API (called from HTML) ─────────────────────────────────────

  window.startCamera = async function () {
    btnStart().disabled = true;
    setLoading("Requesting camera access...");

    try {
      _stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      const v = video();
      v.srcObject = _stream;
      await v.play();

      // Resize landmark canvas to match video
      v.addEventListener("loadedmetadata", () => {
        const cvs = lmCanvas();
        cvs.width = v.videoWidth;
        cvs.height = v.videoHeight;
      });

      btnStart().disabled = true;
      btnStop().disabled  = false;
      scanLine().classList.add("active");
      _setDetectingState(true);
      _startDetection();

    } catch (err) {
      showError(`Camera error: ${err.message}`);
      btnStart().disabled = false;
    } finally {
      hideLoading();
    }
  };

  window.stopDetection = function () {
    _stopDetection();
    if (_stream) {
      _stream.getTracks().forEach(t => t.stop());
      _stream = null;
    }
    video().srcObject = null;
    btnStart().disabled = false;
    btnStop().disabled  = true;
    scanLine().classList.remove("active");
    _setDetectingState(false);
    _clearLandmarkCanvas();
  };

  window.speakLastDetection = async function () {
    if (!_lastLabel) return;
    await _speakSign(_lastLabel);
  };

  // ─── Detection loop ─────────────────────────────────────────────────────

  function _startDetection() {
    if (_detectTimer) clearInterval(_detectTimer);
    _detectTimer = setInterval(_runDetection, DETECT_INTERVAL);
  }

  function _stopDetection() {
    if (_detectTimer) {
      clearInterval(_detectTimer);
      _detectTimer = null;
    }
    _isDetecting = false;
  }

  async function _runDetection() {
    if (_isDetecting) return;   // skip if previous call still pending
    const v = video();
    if (!v || !v.srcObject || v.readyState < 2) return;

    _isDetecting = true;
    try {
      const frameB64 = _captureFrame(v);
      const res = await fetch(`${window.API_BASE}/api/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frameB64, session_id: _sessionId }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      _handleDetectionResult(data);

    } catch (err) {
      console.warn("Detection error:", err);
    } finally {
      _isDetecting = false;
    }
  }

  function _captureFrame(videoEl) {
    const tmp = document.createElement("canvas");
    tmp.width  = videoEl.videoWidth  || 640;
    tmp.height = videoEl.videoHeight || 480;
    tmp.getContext("2d").drawImage(videoEl, 0, 0);
    return tmp.toDataURL("image/jpeg", 0.7).split(",")[1];
  }

  function _handleDetectionResult(data) {
    const label      = data.label;
    const confidence = data.confidence || 0;
    const landmarks  = data.landmarks  || [];

    // Update confidence bar
    const pct = Math.round(confidence * 100);
    const bar = confBar();
    bar.style.width = `${pct}%`;
    bar.className = "confidence-bar" + (pct >= 80 ? " high" : pct >= 50 ? "" : " low");
    confText().textContent = `${pct}%`;

    if (label && confidence >= MIN_CONFIDENCE) {
      // Update detection label
      detectionLabel().textContent = label.replace(/_/g, " ").toUpperCase();
      _typewriterUpdate(detectedText(), label.replace(/_/g, " "));
      _lastLabel = label;

      // Draw landmarks
      if (landmarks.length > 0) _drawLandmarks(landmarks);

      // Auto-speak with debounce
      const now = Date.now();
      if (label !== _lastLabel || now - _lastSpeakTs > DEBOUNCE_SAME) {
        _lastSpeakTs = now;
        _speakSign(label);
        _addPatientChatMessage(label.replace(/_/g, " "), pct);
      }
    } else {
      detectionLabel().textContent = "Detecting...";
      if (!label) _clearLandmarkCanvas();
    }
  }

  // ─── Landmark drawing ───────────────────────────────────────────────────

  // Hand skeleton connections (MediaPipe-style)
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],      // thumb
    [0,5],[5,6],[6,7],[7,8],      // index
    [0,9],[9,10],[10,11],[11,12], // middle
    [0,13],[13,14],[14,15],[15,16],// ring
    [0,17],[17,18],[18,19],[19,20],// pinky
    [5,9],[9,13],[13,17],          // palm
  ];

  function _drawLandmarks(landmarks) {
    const cvs = lmCanvas();
    const ctx = cvs.getContext("2d");
    const W = cvs.width;
    const H = cvs.height;

    ctx.clearRect(0, 0, W, H);

    if (!landmarks || landmarks.length < 21) return;

    // Convert normalised [0,1] coords to canvas pixels
    const pts = landmarks.map(([x, y]) => [x * W, y * H]);

    // Draw connections
    ctx.strokeStyle = "rgba(0, 212, 255, 0.7)";
    ctx.lineWidth = 2;
    for (const [a, b] of CONNECTIONS) {
      if (pts[a] && pts[b]) {
        ctx.beginPath();
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
        ctx.stroke();
      }
    }

    // Draw landmark dots
    landmarks.forEach(([x, y], i) => {
      const px = x * W;
      const py = y * H;
      // Fingertips (4,8,12,16,20) in green; rest in blue
      const isTip = [4, 8, 12, 16, 20].includes(i);
      ctx.beginPath();
      ctx.arc(px, py, isTip ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = isTip ? "rgba(0,255,157,0.9)" : "rgba(0,212,255,0.9)";
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  function _clearLandmarkCanvas() {
    const cvs = lmCanvas();
    if (cvs) cvs.getContext("2d").clearRect(0, 0, cvs.width, cvs.height);
  }

  // ─── TTS ───────────────────────────────────────────────────────────────

  async function _speakSign(label) {
    try {
      const lang = window.currentLang || "en";
      const res = await fetch(`${window.API_BASE}/api/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: label.replace(/_/g, " "), lang }),
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play().catch(() => {});
    } catch (err) {
      console.warn("TTS error:", err);
    }
  }

  // ─── Chat history ──────────────────────────────────────────────────────

  function _addPatientChatMessage(text, confidence) {
    const container = document.getElementById("patient-messages");
    if (!container) return;

    const div = document.createElement("div");
    div.className = "chat-message";
    div.innerHTML = `
      <span class="msg-time">${_now()}</span>
      <span class="msg-label">SIGN</span>
      <span class="msg-text">${_escHtml(text)} <small style="color:var(--text-muted)">(${confidence}%)</small></span>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  // ─── UI helpers ────────────────────────────────────────────────────────

  function _setDetectingState(active) {
    const dot   = dotDetect();
    const badge = badgeDetect();
    if (dot)   dot.classList.toggle("active", active);
    if (badge) badge.classList.toggle("active", active);
  }

  let _twTimeout = null;
  function _typewriterUpdate(el, text) {
    if (!el) return;
    if (el.dataset.current === text) return;
    el.dataset.current = text;
    el.textContent = "";
    el.classList.add("typing");
    clearTimeout(_twTimeout);
    let i = 0;
    function step() {
      if (i < text.length) {
        el.textContent += text[i++];
        _twTimeout = setTimeout(step, 40);
      } else {
        el.classList.remove("typing");
      }
    }
    step();
  }

  function _now() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function _escHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

})();
