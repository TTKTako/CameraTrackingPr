from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── Cameras ──────────────────────────────────────────────────────────────
    camera_indices: List[int] = field(default_factory=lambda: [0])
    # Cameras to start in disabled state (can also be toggled at runtime)
    disabled_cameras: List[int] = field(default_factory=list)

    # ── Video file (overrides live cameras when set) ──────────────────────────
    video_path: Optional[str] = None
    video_loop: bool = True           # loop indefinitely until stopped

    # ── Display ───────────────────────────────────────────────────────────────
    window_name: str = "CameraTrackingPose"
    display_width: int = 1280
    display_height: int = 720

    # ── RTMPose model ─────────────────────────────────────────────────────────
    # 'wholebody' uses the pre-configured RTMPose whole-body alias in MMPose 1.x
    # which includes 133 keypoints: 17 body + 6 feet + 68 face + 42 hands
    pose2d_model: str = "wholebody"
    device: str = "cuda:0"

    # ── WebSocket (Unity) ─────────────────────────────────────────────────────
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    # ── Performance ───────────────────────────────────────────────────────────
    target_fps: int = 30
    # Process every Nth frame for inference (0 = every frame)
    # Example: skip_frames=1 runs inference at half the capture FPS
    skip_frames: int = 0

    # ── Confidence threshold for keypoint rendering / bone mapping ────────────
    kp_confidence: float = 0.3

    # ── Pose template matching ────────────────────────────────────────────────
    # Directory containing reference pose images (.png / .jpg)
    pose_template_dir: str = "poseTemplate"
    # Similarity threshold [0–1]; arm-focused weighted score must reach this
    # (arm bones × 0.7 + full-body × 0.3). Lower = easier to trigger.
    pose_match_threshold: float = 0.75
    # Minimum seconds between consecutive pose_match WebSocket events
    pose_match_cooldown: float = 2.0

    # ── 3D isometric rig panel ────────────────────────────────────────────────
    show_isometric_rig: bool = True
    # Pixel size of the isometric rig panel appended to the camera grid
    iso_rig_size: int = 380
