"""
vrm_mapper.py
-------------
Maps RTMPose 2-D keypoints to VRM Quaternions with Jitter Smoothing.
Outputs strict Unity Left-Handed Quaternions.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation
from .pose_estimator import PoseResult

_KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

_UP    = np.array([0.0,  1.0, 0.0])
_DOWN  = np.array([0.0, -1.0, 0.0])
_LEFT  = np.array([-1.0, 0.0, 0.0])
_RIGHT = np.array([1.0,  0.0, 0.0])

def _to3d(p2d: np.ndarray) -> np.ndarray:
    return np.array([p2d[0], -p2d[1], 0.0])

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else _UP.copy()

def _rotation_align(src: np.ndarray, dst: np.ndarray) -> Rotation:
    src, dst = _norm(src), _norm(dst)
    cross = np.cross(src, dst)
    dot = float(np.dot(src, dst))
    if np.linalg.norm(cross) < 1e-6:
        if dot > 0: return Rotation.identity()
        perp = np.cross(src, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6: perp = np.cross(src, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(np.pi * _norm(perp))
    angle = np.arctan2(np.linalg.norm(cross), dot)
    return Rotation.from_rotvec(angle * _norm(cross))

def _to_unity_quat(rot: Rotation) -> List[float]:
    """Passes the pure quaternion. The points are already in Unity Space."""
    q = rot.as_quat() # [x, y, z, w]
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])] # NO negations!

class EMAFilter:
    """Smooths out AI jitter using Exponential Moving Average."""
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.state: Dict[str, np.ndarray] = {}

    def update(self, key: str, value: List[float]) -> List[float]:
        val_arr = np.array(value)
        if key not in self.state:
            self.state[key] = val_arr
            return value
        
        # Slerp approximation for Quaternions using EMA
        self.state[key] = self.alpha * val_arr + (1.0 - self.alpha) * self.state[key]
        self.state[key] /= np.linalg.norm(self.state[key])
        return self.state[key].tolist()

class VRMMapper:
    def __init__(self, confidence: float = 0.3) -> None:
        self._conf = confidence
        self._filter = EMAFilter(alpha=0.4) # Lower = smoother but slower, Higher = faster but jittery

    def map(self, result: PoseResult) -> Dict[str, List[float]]:
        kps = result.keypoints
        scores = result.keypoint_scores
        bones: Dict[str, List[float]] = {}

        # Ensure your __init__ confidence is around 0.5 for MediaPipe
        def pt(name: str) -> Optional[np.ndarray]:
            idx = _KP.get(name)
            if idx is None or scores[idx] < self._conf:
                return None
            
            # --- TRUE 3D FETCH ---
            # Fetch the actual X, Y, and Z depth from MediaPipe
            x, y, z = result.keypoints_3d[idx]
            
            # Convert to Unity's left-handed coordinate system (X right, Y up, Z forward)
            return np.array([x, -y, -z])

        def bone(from_name: str, to_name: str, tpose_dir: np.ndarray, bone_name: str):
            p0, p1 = pt(from_name), pt(to_name)
            if p0 is not None and p1 is not None:
                rot = _rotation_align(tpose_dir, _norm(p1 - p0))
                bones[bone_name] = self._filter.update(bone_name, _to_unity_quat(rot))

        # Core Body Setup
        # Core Body Setup (MIRRORED FOR WEBCAM)
        # We map MediaPipe's 'left' to Unity's 'Right', using the _RIGHT T-pose direction
        bone("left_shoulder", "left_elbow", _RIGHT, "RightUpperArm")
        bone("left_elbow", "left_wrist", _RIGHT, "RightLowerArm")
        
        # We map MediaPipe's 'right' to Unity's 'Left', using the _LEFT T-pose direction
        bone("right_shoulder", "right_elbow", _LEFT, "LeftUpperArm")
        bone("right_elbow", "right_wrist", _LEFT, "LeftLowerArm")
        
        # You should also mirror the legs so your knees don't cross!
        bone("left_hip", "left_knee", _DOWN, "RightUpperLeg")
        bone("left_knee", "left_ankle", _DOWN, "RightLowerLeg")
        bone("right_hip", "right_knee", _DOWN, "LeftUpperLeg")
        bone("right_knee", "right_ankle", _DOWN, "LeftLowerLeg")

        # Spine Math
        ls, rs, lh, rh = pt("left_shoulder"), pt("right_shoulder"), pt("left_hip"), pt("right_hip")
        if all(x is not None for x in [ls, rs, lh, rh]):
            spine_dir = _norm(((ls + rs) * 0.5) - ((lh + rh) * 0.5))
            spine_rot = _to_unity_quat(_rotation_align(_UP, spine_dir))
            smoothed_spine = self._filter.update("Spine", spine_rot)
            bones["Hips"] = smoothed_spine
            bones["Spine"] = smoothed_spine

        # Head Math
        nose = pt("nose")
        if ls is not None and rs is not None and nose is not None:
            head_rot = _to_unity_quat(_rotation_align(_UP, _norm(nose - ((ls + rs) * 0.5))))
            bones["Head"] = self._filter.update("Head", head_rot)

        # --- NEW: Calculate Global Position (Root X, Y, Z) ---
        root_pos = [0.0, 0.0, 0.0]
        lh, rh = pt("left_hip"), pt("right_hip")
        ls, rs = pt("left_shoulder"), pt("right_shoulder")
        
        if lh is not None and rh is not None and ls is not None and rs is not None:
            # 1. Calculate X and Y (Center of the body)
            # Assuming standard webcam width of 640x480. We subtract 320 to center it.
            center_x = ((lh[0] + rh[0] + ls[0] + rs[0]) / 4.0) - 320.0
            center_y = ((lh[1] + rh[1] + ls[1] + rs[1]) / 4.0) - 240.0
            
            # 2. Calculate Z (Depth) using apparent body width
            # The wider the pixels between your shoulders, the closer you are to the camera.
            pixel_width = np.linalg.norm(ls - rs)
            
            # Tune these modifiers to match your room! 
            # reference_width is your shoulder width in pixels when standing at the "zero" point.
            reference_width = 150.0 
            
            # Calculate final positions (scaled down to Unity meters)
            pos_x = center_x * -0.005  # Inverted for Unity left-handedness
            pos_y = center_y * -0.005  # Up/Down crouching
            pos_z = (pixel_width - reference_width) * 0.02 # Forward/Backward
            
            # Smooth the position using our existing EMA filter so it doesn't jitter
            root_pos = self._filter.update("RootPosition", [pos_x, pos_y, pos_z])
            
        bones["RootPosition"] = root_pos # Inject it into the bone dictionary
        
        # Remove any None values that slipped through
        return {k: v for k, v in bones.items() if v is not None}

    