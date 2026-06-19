"""
pose_matcher.py
---------------
Compares live pose keypoints against pre-loaded template images to detect
when the user is performing a specific pose.

Workflow
────────
1. At startup, load all PNG/JPG images from the template directory.
2. Run RTMPose on each template image to extract COCO-17 body keypoints.
3. Normalise each template's keypoints (translation + scale invariant).
4. At runtime, normalise the live pose and compute similarity to each template.
5. Return the best-matching template name and confidence score.

Normalisation
─────────────
• Translate so the hip midpoint (mean of kps[11], kps[12]) is at the origin.
• Scale by torso length (hip midpoint → shoulder midpoint distance).
• This makes the representation invariant to person size and position.

Similarity
──────────
• Compute bone direction vectors for a fixed set of limb pairs.
• Average cosine similarity across all valid bone pairs.
• Score ∈ [0, 1]; 1.0 = perfect match.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pose_estimator import PoseEstimator, PoseResult

# COCO-17 body keypoints used for comparison
_BODY_N = 17

# Bone pairs (parent → child) used for full-body pose comparison
_COMPARE_BONES: List[Tuple[int, int]] = [
    (5,  6),   # shoulder bar
    (5,  7),   # left upper arm
    (7,  9),   # left lower arm
    (6,  8),   # right upper arm
    (8, 10),   # right lower arm
    (5, 11),   # left torso side
    (6, 12),   # right torso side
    (11, 12),  # hip bar
    (11, 13),  # left upper leg
    (13, 15),  # left lower leg
    (12, 14),  # right upper leg
    (14, 16),  # right lower leg
    (0,  5),   # head → left shoulder
    (0,  6),   # head → right shoulder
]

# Arm-only bone pairs for arm-focused matching
_ARM_BONES: List[Tuple[int, int]] = [
    (5,  6),   # shoulder bar
    (5,  7),   # left upper arm
    (7,  9),   # left lower arm
    (6,  8),   # right upper arm
    (8, 10),   # right lower arm
]

# Weight given to arm-only score vs full-body score [0–1]
# 1.0 = arm bones only; 0.0 = full body only
_ARM_FOCUS_WEIGHT: float = 0.7

_SHOULDER_L = 5
_SHOULDER_R = 6
_HIP_L      = 11
_HIP_R      = 12


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(kps: np.ndarray, scores: np.ndarray, conf: float) -> Optional[np.ndarray]:
    """Return scale+translation-invariant (17, 2) normalised keypoints, or None."""
    body_kps    = kps[:_BODY_N]
    body_scores = scores[:_BODY_N]

    needed = [_SHOULDER_L, _SHOULDER_R, _HIP_L, _HIP_R]
    if any(body_scores[i] < conf for i in needed):
        return None

    hip_mid      = (body_kps[_HIP_L]      + body_kps[_HIP_R])      / 2.0
    shoulder_mid = (body_kps[_SHOULDER_L] + body_kps[_SHOULDER_R]) / 2.0
    torso_len    = float(np.linalg.norm(shoulder_mid - hip_mid))

    if torso_len < 1e-6:
        return None

    return (body_kps - hip_mid) / torso_len


def _similarity(
    norm_a: np.ndarray, scores_a: np.ndarray,
    norm_b: np.ndarray, scores_b: np.ndarray,
    conf: float,
    bones: Optional[List[Tuple[int, int]]] = None,
) -> float:
    """Average cosine similarity of bone direction vectors, mapped to [0, 1]."""
    if bones is None:
        bones = _COMPARE_BONES
    sims: List[float] = []
    for j1, j2 in bones:
        if (scores_a[j1] < conf or scores_a[j2] < conf
                or scores_b[j1] < conf or scores_b[j2] < conf):
            continue
        va = norm_a[j2] - norm_a[j1]
        vb = norm_b[j2] - norm_b[j1]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-6 or nb < 1e-6:
            continue
        # cos ∈ [-1, 1]  →  map to [0, 1]
        cos_sim = float(np.dot(va / na, vb / nb))
        sims.append((cos_sim + 1.0) / 2.0)
    return float(np.mean(sims)) if sims else 0.0


# ── Data class ────────────────────────────────────────────────────────────────

class _PoseTemplate:
    def __init__(self, name: str, norm_kps: np.ndarray, scores: np.ndarray) -> None:
        self.name     = name
        self.norm_kps = norm_kps   # (17, 2) normalised
        self.scores   = scores     # (17,)


# ── Main class ────────────────────────────────────────────────────────────────

class PoseTemplateMatcher:
    """Load pose template images once and match live poses against them."""

    def __init__(
        self,
        template_dir: str,
        estimator: PoseEstimator,
        match_threshold: float = 0.75,
        conf: float = 0.3,
    ) -> None:
        self._threshold = match_threshold
        self._conf      = conf
        self._templates: List[_PoseTemplate] = []
        # Cached scores from the most recent match() call: {template_name: score}
        self._last_all_scores: Dict[str, float] = {}
        # Periodic debug logging (print best score even when below threshold)
        self._last_debug_print: float = 0.0
        self._load_templates(template_dir, estimator)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_templates(self, template_dir: str, estimator: PoseEstimator) -> None:
        path = Path(template_dir)
        if not path.exists():
            print(f"[PoseMatcher] Template directory not found: {template_dir}")
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        images = sorted(f for f in path.iterdir() if f.suffix.lower() in valid_exts)
        print(f"[PoseMatcher] Loading {len(images)} template(s) from '{template_dir}' …")

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[PoseMatcher]   SKIP (unreadable): {img_path.name}")
                continue

            results = estimator.estimate(frame)
            if not results:
                print(f"[PoseMatcher]   SKIP (no pose detected): {img_path.name}")
                continue

            best     = PoseEstimator.get_closest_person(results)
            norm_kps = _normalise(best.keypoints, best.keypoint_scores, self._conf)
            if norm_kps is None:
                print(f"[PoseMatcher]   SKIP (insufficient keypoints): {img_path.name}")
                continue

            self._templates.append(_PoseTemplate(
                name     = img_path.stem,
                norm_kps = norm_kps,
                scores   = best.keypoint_scores[:_BODY_N].copy(),
            ))
            print(f"[PoseMatcher]   Loaded: {img_path.name}")

        print(f"[PoseMatcher] {len(self._templates)} template(s) ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def template_names(self) -> List[str]:
        return [t.name for t in self._templates]

    @property
    def has_templates(self) -> bool:
        return bool(self._templates)

    @property
    def last_all_scores(self) -> Dict[str, float]:
        """Per-template similarity scores from the last match() call."""
        return dict(self._last_all_scores)

    def match(self, result: PoseResult) -> Optional[Tuple[str, float]]:
        """
        Compare a live PoseResult against all templates.

        Returns (template_name, score) if the best score exceeds the threshold,
        otherwise None.  Always updates last_all_scores with per-template values.
        """
        if not self._templates:
            return None

        norm_kps = _normalise(result.keypoints, result.keypoint_scores, self._conf)
        if norm_kps is None:
            # Normalisation requires shoulders + hips above conf — log when this fails
            now = time.perf_counter()
            if now - self._last_debug_print >= 3.0:
                needed_scores = [
                    result.keypoint_scores[i]
                    for i in [_SHOULDER_L, _SHOULDER_R, _HIP_L, _HIP_R]
                ]
                print(
                    f"[PoseMatcher] Cannot normalise — shoulder/hip scores: "
                    f"{[f'{s:.2f}' for s in needed_scores]}  "
                    f"(need >= {self._conf:.2f})"
                )
                self._last_debug_print = now
            self._last_all_scores = {}
            return None

        live_scores = result.keypoint_scores[:_BODY_N]

        best_name  = ""
        best_score = 0.0
        self._last_all_scores = {}
        for tpl in self._templates:
            # Arm-focused score: weighted blend of arm-only and full-body similarity
            arm_score  = _similarity(norm_kps, live_scores,
                                     tpl.norm_kps, tpl.scores,
                                     self._conf, bones=_ARM_BONES)
            body_score = _similarity(norm_kps, live_scores,
                                     tpl.norm_kps, tpl.scores,
                                     self._conf, bones=_COMPARE_BONES)
            score = arm_score * _ARM_FOCUS_WEIGHT + body_score * (1.0 - _ARM_FOCUS_WEIGHT)
            self._last_all_scores[tpl.name] = score
            if score > best_score:
                best_score = score
                best_name  = tpl.name

        # Periodic debug: always print best score so threshold tuning is easy
        now = time.perf_counter()
        if now - self._last_debug_print >= 3.0:
            if self._templates:
                print(
                    f"[PoseMatcher] Best match: '{best_name}'  "
                    f"score={best_score:.3f}  threshold={self._threshold:.3f}  "
                    f"{'MATCHED' if best_score >= self._threshold else 'no match'}"
                )
            self._last_debug_print = now

        if best_score >= self._threshold:
            return (best_name, best_score)
        return None
