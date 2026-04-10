"""
SignBridge AI — NLP Mapper (nlp_mapper.py)

Converts free-form English text into a sequence of ISL sign tokens
and loads the corresponding keypoint animation data.

Pipeline:
  Input text → tokenise → ISL vocabulary lookup → return token list + keypoints
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISL vocabulary: English word/phrase → sign token name
# ---------------------------------------------------------------------------

ISL_VOCAB: Dict[str, str] = {
    # Core medical
    "help": "help",
    "pain": "pain",
    "hurt": "pain",
    "hurts": "pain",
    "ache": "pain",
    "headache": "headache",
    "head ache": "headache",
    "head pain": "headache",
    "stomach pain": "stomach_pain",
    "stomach ache": "stomach_pain",
    "tummy ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "abdomen pain": "stomach_pain",
    "chest pain": "chest_pain",
    "heart pain": "chest_pain",
    "emergency": "emergency",
    "urgent": "emergency",
    "stop": "stop",
    "halt": "stop",
    "no": "no",
    "yes": "yes",
    "okay": "yes",
    "ok": "yes",
    "doctor": "call_doctor",
    "call doctor": "call_doctor",
    "nurse": "call_doctor",
    "call nurse": "call_doctor",
    "no pain": "no_pain",
    "better": "no_pain",
    "thank you": "thank_you",
    "thanks": "thank_you",
    "water": "water",
    "drink": "water",
    "thirsty": "water",
    "medicine": "medicine",
    "medication": "medicine",
    "pill": "medicine",
    "drug": "medicine",
    # Body parts
    "head": "head",
    "chest": "chest",
    "stomach": "stomach",
    "back": "back",
    "hand": "hand",
    "leg": "leg",
    "foot": "leg",
    "feet": "leg",
    "arm": "hand",
}

# Multi-word phrases sorted by length (longest first) for greedy matching
_PHRASES = sorted(
    [(k, v) for k, v in ISL_VOCAB.items() if " " in k],
    key=lambda x: -len(x[0]),
)

# Single-word mappings
_WORDS = {k: v for k, v in ISL_VOCAB.items() if " " not in k}


class NLPMapper:
    """
    Maps input text to a sequence of ISL sign tokens and loads
    keypoint animation sequences from JSON files.
    """

    def __init__(self, keypoints_dir: str):
        self._keypoints_dir = keypoints_dir
        self._keypoint_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_phrase(self, text: str) -> Dict[str, Any]:
        """
        Map a phrase to ISL tokens and return keypoint sequences.

        Args:
            text: Free-form input text (e.g. "I have chest pain").

        Returns:
            {
              "tokens": ["chest", "pain"],
              "sequences": {
                "chest": { ...keypoint data... },
                "pain":  { ...keypoint data... },
              },
              "unknown_tokens": ["i", "have"]
            }
        """
        tokens, unknown = self._tokenise(text)
        sequences = {}
        for token in tokens:
            kp = self._load_keypoints(token)
            if kp:
                sequences[token] = kp

        return {
            "tokens": tokens,
            "sequences": sequences,
            "unknown_tokens": unknown,
        }

    def map_word(self, word: str) -> Optional[str]:
        """Map a single word to its ISL sign token."""
        clean = word.strip().lower()
        return _WORDS.get(clean)

    def list_supported_signs(self) -> List[str]:
        """Return all supported sign tokens."""
        return sorted(set(ISL_VOCAB.values()))

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> tuple:
        """
        Greedy tokenisation: multi-word phrases matched first,
        then individual words.
        Returns (token_list, unknown_word_list).
        """
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens: List[str] = []
        unknown: List[str] = []
        remaining = text

        while remaining:
            matched = False

            # Try multi-word phrases first
            for phrase, sign_token in _PHRASES:
                if remaining.startswith(phrase):
                    tokens.append(sign_token)
                    remaining = remaining[len(phrase):].strip()
                    matched = True
                    break

            if matched:
                continue

            # Split off next word
            parts = remaining.split(" ", 1)
            word = parts[0]
            remaining = parts[1] if len(parts) > 1 else ""

            if word in _WORDS:
                sign_token = _WORDS[word]
                # Avoid consecutive duplicates
                if not tokens or tokens[-1] != sign_token:
                    tokens.append(sign_token)
            elif word:
                unknown.append(word)

        return tokens, unknown

    # ------------------------------------------------------------------
    # Keypoint loading
    # ------------------------------------------------------------------

    # Valid token pattern: lowercase letters, digits, underscores only
    _TOKEN_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')

    def _load_keypoints(self, token: str) -> Optional[Dict]:
        """Load keypoint JSON for a sign token. Cached after first load."""
        if token in self._keypoint_cache:
            return self._keypoint_cache[token]

        # Validate token to prevent path traversal
        if not self._TOKEN_RE.match(token):
            logger.warning(f"Invalid token rejected: '{token}'")
            return None

        # Build path using only the validated token — safe against path traversal
        json_filename = token + ".json"
        json_path = os.path.join(self._keypoints_dir, json_filename)

        # Ensure the resolved path is inside the keypoints directory
        keypoints_dir_real = os.path.realpath(self._keypoints_dir)
        json_path_real = os.path.realpath(json_path)
        if not json_path_real.startswith(keypoints_dir_real + os.sep):
            logger.warning(f"Path traversal attempt blocked for token: '{token}'")
            return None

        if not os.path.exists(json_path_real):
            logger.warning(f"Keypoint file not found: {json_path_real}")
            return None

        try:
            with open(json_path_real, "r") as f:
                data = json.load(f)
            self._keypoint_cache[token] = data
            return data
        except Exception as exc:
            logger.error(f"Failed to load keypoints for '{token}': {exc}")
            return None
