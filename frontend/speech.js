/**
 * SignBridge AI — speech.js
 * Module 2: Doctor Speech Recognition → ISL Avatar
 *
 * - Uses Web Speech API for real-time STT
 * - Sends recognised text to /api/mapper to get ISL tokens + keypoints
 * - Feeds keypoint sequences to avatarModule for animation
 * - Adds messages to doctor chat history
 */

(function () {
  "use strict";

  // ─── Web Speech API support check ──────────────────────────────────────
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  let _recognition  = null;
  let _isListening  = false;
  let _apiBase      = "http://localhost:5000";
  let _lang         = "en-US";

  const LANG_BCP47 = { en: "en-US", ta: "ta-IN", hi: "hi-IN" };

  // ─── Public module object ──────────────────────────────────────────────
  window.speechModule = {
    init,
    setLang,
    startListening,
    stopListening,
    submitText,
  };

  // ─── DOM refs ──────────────────────────────────────────────────────────
  const micBtn     = () => document.getElementById("mic-btn");
  const speechText = () => document.getElementById("speech-text");
  const dotListen  = () => document.getElementById("dot-listen");
  const badgeListen= () => document.getElementById("badge-listen");
  const textInput  = () => document.getElementById("doctor-text-input");

  // ─── Initialisation ────────────────────────────────────────────────────

  function init(apiBase, lang) {
    _apiBase = apiBase;
    setLang(lang);

    if (!SpeechRecognition) {
      console.warn("Web Speech API not supported in this browser.");
      if (micBtn()) {
        micBtn().title = "Speech recognition not supported";
        micBtn().style.opacity = "0.4";
      }
      return;
    }

    _recognition = new SpeechRecognition();
    _recognition.continuous     = false;
    _recognition.interimResults = true;
    _recognition.lang           = _lang;

    _recognition.onstart = () => {
      _isListening = true;
      micBtn()?.classList.add("listening");
      _setListeningState(true);
      _updateSpeechText("Listening...", true);
    };

    _recognition.onend = () => {
      _isListening = false;
      micBtn()?.classList.remove("listening");
      _setListeningState(false);
    };

    _recognition.onerror = (e) => {
      _isListening = false;
      micBtn()?.classList.remove("listening");
      _setListeningState(false);
      if (e.error !== "no-speech") {
        console.warn("Speech error:", e.error);
        showError(`Microphone error: ${e.error}`);
      }
    };

    _recognition.onresult = (e) => {
      let interim = "";
      let final   = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t;
        else interim += t;
      }

      if (interim) _updateSpeechText(interim, false);
      if (final)   _onFinalSpeech(final.trim());
    };
  }

  function setLang(lang) {
    _lang = LANG_BCP47[lang] || "en-US";
    if (_recognition) _recognition.lang = _lang;
  }

  // ─── Toggle microphone ──────────────────────────────────────────────────

  window.toggleMicrophone = function () {
    if (_isListening) stopListening();
    else              startListening();
  };

  function startListening() {
    if (!_recognition) {
      showError("Speech recognition is not available in this browser.");
      return;
    }
    if (_isListening) return;
    try {
      _recognition.start();
    } catch (e) {
      console.warn("Recognition start error:", e);
    }
  }

  function stopListening() {
    if (_recognition && _isListening) {
      _recognition.stop();
    }
  }

  // ─── Submit typed text ─────────────────────────────────────────────────

  window.submitDoctorText = function () {
    const input = textInput();
    if (!input) return;
    const text = input.value.trim();
    if (text) {
      input.value = "";
      submitText(text);
    }
  };

  window.handleDoctorInput = function (event) {
    if (event.key === "Enter") window.submitDoctorText();
  };

  async function submitText(text) {
    _updateSpeechText(text, false);
    _addDoctorChatMessage(text);
    await _mapToISL(text);
  }

  // ─── Internal: final speech handling ───────────────────────────────────

  async function _onFinalSpeech(text) {
    if (!text) return;
    _updateSpeechText(text, false);
    _addDoctorChatMessage(text);
    await _mapToISL(text);
  }

  // ─── Map text to ISL and animate avatar ────────────────────────────────

  async function _mapToISL(text) {
    try {
      const res = await fetch(`${_apiBase}/api/mapper`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.tokens && data.tokens.length > 0) {
        // Update avatar sign label
        const labelEl = document.getElementById("avatar-sign-label");
        if (labelEl) {
          labelEl.textContent = data.tokens.map(t => t.replace(/_/g, " ")).join(" → ");
        }

        // Feed sequences to avatar module
        if (window.avatarModule && data.sequences) {
          await window.avatarModule.playSequence(data.tokens, data.sequences);
        }
      }
    } catch (err) {
      console.warn("Mapper error:", err);
    }
  }

  // ─── UI helpers ────────────────────────────────────────────────────────

  function _setListeningState(active) {
    const dot   = dotListen();
    const badge = badgeListen();
    if (dot)   dot.classList.toggle("active", active);
    if (badge) badge.classList.toggle("active", active);
  }

  let _twTimer = null;
  function _updateSpeechText(text, interim) {
    const el = speechText();
    if (!el) return;
    clearTimeout(_twTimer);
    if (interim) {
      el.textContent = text;
      el.style.opacity = "0.6";
      return;
    }
    el.style.opacity = "1";
    el.textContent = "";
    el.classList.add("typing");
    let i = 0;
    function step() {
      if (i < text.length) {
        el.textContent += text[i++];
        _twTimer = setTimeout(step, 30);
      } else {
        el.classList.remove("typing");
      }
    }
    step();
  }

  function _addDoctorChatMessage(text) {
    const container = document.getElementById("doctor-messages");
    if (!container) return;
    const div = document.createElement("div");
    div.className = "chat-message";
    div.innerHTML = `
      <span class="msg-time">${_now()}</span>
      <span class="msg-label" style="color:var(--neon-green)">DOCTOR</span>
      <span class="msg-text">${_escHtml(text)}</span>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  function _now() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function _escHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

})();
