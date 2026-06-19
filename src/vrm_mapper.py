"""
vrm_mapper.py
-------------
Maps RTMPose whole-body 2-D keypoints → VRM HumanBodyBones quaternions.

Coordinate convention
─────────────────────
  RTMPose output : image space  (x→right, y→down)
  VRM / Unity    : Y-up right-hand  (x→right, y→up, z→forward)

Conversion: flip y  →  (px, -py, 0)  (z=0, facing camera approximation)

Bone rotations are in world space relative to the T-pose.
Unity / VRM consumers should convert to local space if required.

T-pose reference directions
────────────────────────────
  Spine / Neck / Head : (0, +1, 0)  up
  Left  arm segments  : (-1, 0, 0)  pointing left
  Right arm segments  : (+1, 0, 0)  pointing right
  Both  leg segments  : (0, -1, 0)  pointing down
  Left  fingers       : (-1, 0, 0)
  Right fingers       : (+1, 0, 0)
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .pose_estimator import PoseResult

# ── COCO-17 body keypoint name → index ────────────────────────────────────────
_KP = {
    "nose": 0,
    "left_eye": 1,   "right_eye": 2,
    "left_ear": 3,   "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

# Hand keypoint base offsets in the 133-keypoint array
_LH_BASE = 91   # left  hand  kps 91-111
_RH_BASE = 112  # right hand  kps 112-132

# T-pose unit vectors
_UP    = np.array([0.0,  1.0, 0.0])
_DOWN  = np.array([0.0, -1.0, 0.0])
_LEFT  = np.array([-1.0, 0.0, 0.0])
_RIGHT = np.array([1.0,  0.0, 0.0])

# Finger names and their start-offset from the hand base keypoint
_FINGER_NAMES   = ["Thumb", "Index", "Middle", "Ring", "Little"]
_FINGER_OFFSETS = [1, 5, 9, 13, 17]   # offset to metacarpal kp within hand


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to3d(p2d: np.ndarray) -> np.ndarray:
    """Convert 2-D image-space point to 3-D Y-up (z=0)."""
    return np.array([p2d[0], -p2d[1], 0.0])


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else _UP.copy()


def _rotation_align(src: np.ndarray, dst: np.ndarray) -> Rotation:
    """Return the minimal rotation that maps unit vector src → dst."""
    src, dst = _norm(src), _norm(dst)
    cross = np.cross(src, dst)
    dot   = float(np.dot(src, dst))

    if np.linalg.norm(cross) < 1e-6:
        if dot > 0:
            return Rotation.identity()
        # 180° rotation — find any perpendicular axis
        perp = np.cross(src, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(src, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(np.pi * _norm(perp))

    angle = np.arctan2(np.linalg.norm(cross), dot)
    return Rotation.from_rotvec(angle * _norm(cross))


# ── Main mapper ───────────────────────────────────────────────────────────────

class VRMMapper:
    """
    Converts a PoseResult (133 whole-body keypoints) into a dict of
    VRM HumanBodyBones → quaternion [x, y, z, w].
    """

    def __init__(self, confidence: float = 0.3) -> None:
        self._conf = confidence

    def map(self, result: PoseResult) -> Dict[str, List[float]]:
        kps    = result.keypoints         # (K, 2)
        scores = result.keypoint_scores   # (K,)
        bones: Dict[str, List[float]] = {}

        def pt(name: str) -> Optional[np.ndarray]:
            idx = _KP.get(name)
            if idx is None or scores[idx] < self._conf:
                return None
            return _to3d(kps[idx])

        def bone(from_name: str, to_name: str,
                 tpose_dir: np.ndarray) -> Optional[List[float]]:
            p0, p1 = pt(from_name), pt(to_name)
            if p0 is None or p1 is None:
                return None
            current = _norm(p1 - p0)
            return _rotation_align(tpose_dir, current).as_quat().tolist()

        # ── Hips / Spine / Chest ─────────────────────────────────────────────
        ls, rs = pt("left_shoulder"), pt("right_shoulder")
        lh, rh = pt("left_hip"),      pt("right_hip")

        if lh is not None and rh is not None and ls is not None and rs is not None:
            hip_center   = (lh + rh) * 0.5
            shld_center  = (ls + rs) * 0.5
            spine_dir    = _norm(shld_center - hip_center)
            hip_rot      = _rotation_align(_UP, spine_dir).as_quat().tolist()
            bones["Hips"]       = hip_rot
            bones["Spine"]      = hip_rot
            bones["Chest"]      = hip_rot
            bones["UpperChest"] = hip_rot

        # ── Neck / Head ──────────────────────────────────────────────────────
        nose = pt("nose")
        if ls is not None and rs is not None and nose is not None:
            neck = (ls + rs) * 0.5
            head_dir = _norm(nose - neck)
            neck_rot = _rotation_align(_UP, head_dir).as_quat().tolist()
            bones["Neck"] = neck_rot
            bones["Head"] = neck_rot

        # ── Shoulders (clavicle offset) ──────────────────────────────────────
        if ls is not None and rs is not None:
            c  = (ls + rs) * 0.5
            bones["LeftShoulder"]  = _rotation_align(_LEFT,  _norm(ls - c)).as_quat().tolist()
            bones["RightShoulder"] = _rotation_align(_RIGHT, _norm(rs - c)).as_quat().tolist()

        # ── Arms ─────────────────────────────────────────────────────────────
        r = bone("left_shoulder",  "left_elbow",   _LEFT);  bones["LeftUpperArm"]  = r or bones.get("LeftUpperArm", _id_quat())
        r = bone("left_elbow",     "left_wrist",   _LEFT);  bones["LeftLowerArm"]  = r or bones.get("LeftLowerArm", _id_quat())
        r = bone("right_shoulder", "right_elbow",  _RIGHT); bones["RightUpperArm"] = r or bones.get("RightUpperArm", _id_quat())
        r = bone("right_elbow",    "right_wrist",  _RIGHT); bones["RightLowerArm"] = r or bones.get("RightLowerArm", _id_quat())

        # ── Legs ─────────────────────────────────────────────────────────────
        r = bone("left_hip",   "left_knee",   _DOWN); bones["LeftUpperLeg"]  = r or _id_quat()
        r = bone("left_knee",  "left_ankle",  _DOWN); bones["LeftLowerLeg"]  = r or _id_quat()
        r = bone("right_hip",  "right_knee",  _DOWN); bones["RightUpperLeg"] = r or _id_quat()
        r = bone("right_knee", "right_ankle", _DOWN); bones["RightLowerLeg"] = r or _id_quat()

        # ── Feet (approximate) ────────────────────────────────────────────────
        la, ra = pt("left_ankle"), pt("right_ankle")
        if la is not None:
            bones["LeftFoot"]  = _rotation_align(_RIGHT, np.array([1.0, 0.0, 0.0])).as_quat().tolist()
        if ra is not None:
            bones["RightFoot"] = _rotation_align(_LEFT,  np.array([-1.0, 0.0, 0.0])).as_quat().tolist()

        # ── Hands / Fingers ──────────────────────────────────────────────────
        if kps.shape[0] >= 133:
            self._map_hand(kps, scores, bones, _LH_BASE, "Left")
            self._map_hand(kps, scores, bones, _RH_BASE, "Right")

        # Remove any None values that slipped through
        return {k: v for k, v in bones.items() if v is not None}

    def _map_hand(
        self,
        kps: np.ndarray,
        scores: np.ndarray,
        bones: Dict[str, List[float]],
        base: int,
        side: str,
    ) -> None:
        tpose_dir = _LEFT if side == "Left" else _RIGHT
        segment_names = ["Proximal", "Intermediate", "Distal"]

        for fname, fstart in zip(_FINGER_NAMES, _FINGER_OFFSETS):
            for si, sname in enumerate(segment_names):
                i0 = base + fstart + si
                i1 = base + fstart + si + 1
                if i1 >= kps.shape[0]:
                    continue
                if scores[i0] < self._conf or scores[i1] < self._conf:
                    continue
                p0 = _to3d(kps[i0])
                p1 = _to3d(kps[i1])
                current = _norm(p1 - p0)
                rot = _rotation_align(tpose_dir, current).as_quat().tolist()
                bones[f"{side}{fname}{sname}"] = rot


def _id_quat() -> List[float]:
    """Identity quaternion [x, y, z, w]."""
    return [0.0, 0.0, 0.0, 1.0]
