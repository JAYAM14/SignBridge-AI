"""
SignBridge AI — Text-to-Speech Engine (tts_engine.py)

Features:
  - gTTS synthesis for English (en), Tamil (ta), Hindi (hi)
  - File-based cache: same text+lang → cached MP3, never re-generated
  - Thread-safe
"""

import os
import hashlib
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    logger.warning("gTTS not installed — TTS will return silent audio.")


# ---------------------------------------------------------------------------
# Supported language codes
# ---------------------------------------------------------------------------

LANG_MAP = {
    "en": "en",
    "ta": "ta",
    "hi": "hi",
}

# Pre-defined phrases in different languages for common hospital signs
PHRASE_TRANSLATIONS = {
    "help": {"en": "Help", "ta": "உதவி", "hi": "मदद करो"},
    "pain": {"en": "I have pain", "ta": "என்னால் வலி உள்ளது", "hi": "मुझे दर्द है"},
    "headache": {"en": "I have a headache", "ta": "தலைவலி உள்ளது", "hi": "सिरदर्द है"},
    "stomach_pain": {"en": "Stomach pain", "ta": "வயிற்று வலி", "hi": "पेट दर्द"},
    "chest_pain": {"en": "Chest pain", "ta": "மார்பு வலி", "hi": "सीने में दर्द"},
    "emergency": {"en": "Emergency", "ta": "அவசரநிலை", "hi": "आपातकाल"},
    "stop": {"en": "Stop", "ta": "நிறுத்துங்கள்", "hi": "रुको"},
    "call_doctor": {"en": "Call the doctor", "ta": "மருத்துவரை அழையுங்கள்", "hi": "डॉक्टर को बुलाओ"},
    "no_pain": {"en": "No pain", "ta": "வலி இல்லை", "hi": "दर्द नहीं"},
    "yes": {"en": "Yes", "ta": "ஆம்", "hi": "हाँ"},
    "no": {"en": "No", "ta": "இல்லை", "hi": "नहीं"},
    "thank_you": {"en": "Thank you", "ta": "நன்றி", "hi": "धन्यवाद"},
    "water": {"en": "I need water", "ta": "தண்ணீர் வேண்டும்", "hi": "पानी चाहिए"},
    "medicine": {"en": "I need medicine", "ta": "மருந்து வேண்டும்", "hi": "दवाई चाहिए"},
}


class TTSEngine:
    """
    Converts text to MP3 audio using gTTS with file-based caching.
    """

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, lang: str = "en") -> str:
        """
        Convert text to speech.

        Args:
            text: Input text to speak.
            lang: Language code — 'en', 'ta', or 'hi'.

        Returns:
            Absolute path to MP3 audio file.
        """
        lang = LANG_MAP.get(lang, "en")
        cache_path = self._cache_path(text, lang)

        with self._lock:
            if os.path.exists(cache_path):
                return cache_path
            return self._generate(text, lang, cache_path)

    def synthesize_sign(self, sign_label: str, lang: str = "en") -> str:
        """
        Synthesize speech for a known sign label using pre-defined translations.
        Falls back to synthesizing the raw label if not in dictionary.
        """
        translations = PHRASE_TRANSLATIONS.get(sign_label.lower(), {})
        text = translations.get(lang, sign_label.replace("_", " "))
        return self.synthesize(text, lang=lang)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, text: str, lang: str) -> str:
        key = hashlib.md5(f"{lang}:{text}".encode()).hexdigest()
        return os.path.join(self._cache_dir, f"{key}.mp3")

    def _generate(self, text: str, lang: str, output_path: str) -> str:
        if not _GTTS_AVAILABLE:
            return self._generate_silent(output_path)

        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(output_path)
            logger.info(f"TTS generated: '{text}' [{lang}] → {output_path}")
            return output_path
        except Exception as exc:
            logger.error(f"gTTS error: {exc}")
            return self._generate_silent(output_path)

    def _generate_silent(self, output_path: str) -> str:
        """Write a minimal valid (silent) MP3 when gTTS is unavailable."""
        # Minimal MP3 header bytes — ~0.1s of silence
        silent_mp3 = bytes([
            0xFF, 0xFB, 0x90, 0x00,  # MPEG sync + header
            0x00, 0x00, 0x00, 0x00,
        ])
        with open(output_path, "wb") as f:
            f.write(silent_mp3)
        return output_path

    def clear_cache(self):
        """Remove all cached audio files."""
        with self._lock:
            for fname in os.listdir(self._cache_dir):
                if fname.endswith(".mp3"):
                    os.remove(os.path.join(self._cache_dir, fname))
