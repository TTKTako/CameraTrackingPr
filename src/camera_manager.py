"""
camera_manager.py
-----------------
Manages multiple OpenCV camera captures and/or a video file source.

Key capabilities:
  • Open any number of camera indices (0..n)
  • Enable / disable individual cameras at any time (runtime toggle)
  • Load a video file that replaces live cameras
  • Video loops indefinitely until manually stopped
"""
import platform
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config

# --- NEW HELPER FUNCTION ---
def _open_camera(idx: int) -> cv2.VideoCapture:
    """Uses DirectShow on Windows to allow multiple webcams simultaneously."""
    if platform.system() == "Windows":
        return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    return cv2.VideoCapture(idx)

def scan_cameras(max_index: int = 8) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for i in range(max_index + 1):
        cap = _open_camera(i) # <-- CHANGED THIS LINE
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            desc = f"{i} — {w}×{h}" if w and h else f"{i} — unknown"
            found.append((i, desc))
            cap.release()
    return found


def scan_cameras(max_index: int = 8) -> List[Tuple[int, str]]:
    """
    Probe camera indices 0..max_index and return a list of
    (index, description) for every device that opens successfully.

    Description format: "0 — 1920×1080"  (or "0 — unknown" if props unavailable)
    This call can be slow (~0.3 s per index) so run it in a thread.
    """
    found: List[Tuple[int, str]] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            desc = f"{i} — {w}×{h}" if w and h else f"{i} — unknown"
            found.append((i, desc))
            cap.release()
    return found


@dataclass
class _CameraSource:
    index: int
    label: str
    cap: Optional[cv2.VideoCapture]
    enabled: bool
    last_frame: Optional[np.ndarray] = None


class CameraManager:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._cameras: Dict[int, _CameraSource] = {}
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def open_cameras(self) -> None:
        """Open every camera listed in config.camera_indices."""
        for idx in self._cfg.camera_indices:
            cap = _open_camera(idx) # Using the DirectShow fix from earlier
            
            if cap and cap.isOpened():
                # THE MULTI-CAM FIX: Force lower resolution so the USB Controller doesn't crash!
                # 640x480 is plenty of pixels for MediaPipe to track perfectly.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                print(f"[CameraManager] Warning: Camera {idx} could not be opened.")
                cap = None
                
            self._cameras[idx] = _CameraSource(
                index=idx,
                label=f"Camera {idx}",
                cap=cap,
                enabled=(idx not in self._cfg.disabled_cameras),
            )
            state = "ENABLED" if self._cameras[idx].enabled else "DISABLED"
            print(f"[CameraManager] Camera {idx} registered — {state}")

    def load_video(self, path: str) -> None:
        """Load a video file. When loaded, all enabled-camera slots show the video."""
        with self._lock:
            if self._video_cap:
                self._video_cap.release()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(f"[CameraManager] Cannot open video: {path}")
            self._video_cap = cap
            print(f"[CameraManager] Video loaded: {path}")

    def release(self) -> None:
        """Release all resources."""
        for src in self._cameras.values():
            if src.cap:
                src.cap.release()
        if self._video_cap:
            self._video_cap.release()

    # ── Camera control ────────────────────────────────────────────────────────

    def enable_camera(self, index: int) -> None:
        if index in self._cameras:
            self._cameras[index].enabled = True
            print(f"[CameraManager] Camera {index} → ENABLED")

    def disable_camera(self, index: int) -> None:
        if index in self._cameras:
            self._cameras[index].enabled = False
            print(f"[CameraManager] Camera {index} → DISABLED")

    def toggle_camera(self, index: int) -> bool:
        """Toggle a camera's enabled state. Returns the new state."""
        if index in self._cameras:
            new_state = not self._cameras[index].enabled
            self._cameras[index].enabled = new_state
            print(f"[CameraManager] Camera {index} → {'ENABLED' if new_state else 'DISABLED'}")
            return new_state
        return False

    def set_camera_label(self, index: int, label: str) -> None:
        if index in self._cameras:
            self._cameras[index].label = label

    def get_states(self) -> Dict[int, bool]:
        """Return {camera_index: is_enabled} for all registered cameras."""
        return {i: s.enabled for i, s in self._cameras.items()}

    @property
    def active_indices(self) -> List[int]:
        return [i for i, s in self._cameras.items() if s.enabled]

    # ── Frame reading ─────────────────────────────────────────────────────────

    def read_frames(self) -> Dict[int, np.ndarray]:
        """
        Grab one frame per enabled camera (or video file).
        Returns {camera_index: BGR frame}.
        """
        frames: Dict[int, np.ndarray] = {}

        # ── Video file mode ──
        if self._video_cap:
            with self._lock:
                ret, frame = self._video_cap.read()
                if not ret:
                    if self._cfg.video_loop:
                        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self._video_cap.read()
            if ret:
                for idx, src in self._cameras.items():
                    if src.enabled:
                        frames[idx] = frame.copy()
            return frames

        # ── Live camera mode ──
        for idx, src in self._cameras.items():
            if not src.enabled or src.cap is None:
                continue
            ret, frame = src.cap.read()
            if ret:
                src.last_frame = frame
                frames[idx] = frame
            elif src.last_frame is not None:
                # Return last known frame instead of dropping the camera
                frames[idx] = src.last_frame

        return frames

    def get_labels(self) -> Dict[int, str]:
        return {i: s.label for i, s in self._cameras.items()}
