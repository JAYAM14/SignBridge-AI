"""
SignBridge AI — User Calibration (calibration.py)

Stores per-session hand calibration data (wrist anchor offset + scale)
so the gesture detector adapts to each user's hand size.
"""

import logging
import threading
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Sessions expire after 30 minutes of inactivity
SESSION_TIMEOUT_S = 1800


class Calibration:
    """
    In-memory store of per-session calibration data.

    Calibration data is derived from a 30-second baseline capture:
      - offset: mean wrist position (subtracted before normalisation)
      - scale:  ratio of user's hand span to reference span
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, session_id: str, landmarks: List[List[float]]) -> Dict:
        """
        Compute and store calibration parameters from a set of reference
        landmark observations captured during the calibration phase.

        Args:
            session_id: Unique client session identifier.
            landmarks:  List of 21-landmark observations, each a list of
                        [x, y, z] coordinates.
                        Accepts either:
                          - A single observation:  [[x,y,z] × 21]
                          - Multiple observations: [[[x,y,z] × 21] × N]

        Returns:
            {"offset": [...], "scale": float}
        """
        arr = np.array(landmarks, dtype=np.float32)

        # Handle single observation (21, 3)
        if arr.ndim == 2 and arr.shape == (21, 3):
            arr = arr[np.newaxis, ...]  # → (1, 21, 3)

        # Handle flat (63,) input
        if arr.ndim == 1 and arr.shape[0] == 63:
            arr = arr.reshape(1, 21, 3)

        if arr.ndim != 3 or arr.shape[1:] != (21, 3):
            raise ValueError(
                f"Unexpected landmarks shape {arr.shape}. "
                "Expected (N, 21, 3) or (21, 3)."
            )

        # Offset = mean wrist position across all observations
        wrist_positions = arr[:, 0, :]    # (N, 3)
        offset = wrist_positions.mean(axis=0).tolist()

        # Scale = mean wrist-to-middle-MCP distance across all observations
        ref_distances = np.linalg.norm(arr[:, 9, :] - arr[:, 0, :], axis=1)
        mean_dist = float(ref_distances.mean())
        reference_dist = 0.4   # Expected normalised distance for average hand
        scale = reference_dist / mean_dist if mean_dist > 1e-6 else 1.0

        calibration_data = {"offset": offset, "scale": scale}

        with self._lock:
            self._sessions[session_id] = calibration_data

        logger.info(
            f"Calibration stored for session '{session_id}': "
            f"offset={offset}, scale={scale:.4f}"
        )
        return calibration_data

    def get(self, session_id: str) -> Optional[Dict]:
        """Retrieve calibration data for a session. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def clear(self, session_id: str):
        """Remove calibration data for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self):
        """Remove all calibration sessions."""
        with self._lock:
            self._sessions.clear()
