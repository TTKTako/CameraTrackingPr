"""
main.py
-------
Application entry point.

Usage examples
──────────────
  # Single camera (default)
  python -m src.main

  # Two cameras, start with camera 1 disabled
  python -m src.main --cameras 0 1 --disable 1

  # Load a video file (loops forever)
  python -m src.main --video path/to/video.mp4

  # Override device / port
  python -m src.main --device cuda:0 --ws-port 8765 --fps 30

Runtime keyboard shortcuts
───────────────────────────
  Q          : quit
  0 – 9      : toggle the corresponding camera on / off
"""

import argparse
import time
from typing import Dict, List, Optional, Tuple

import cv2

from .config import Config
from .camera_manager import CameraManager
from .pose_estimator import PoseEstimator, PoseResult
from .vrm_mapper import VRMMapper
from .display import DisplayManager
from .websocket_server import PoseWebSocketServer
from .pose_matcher import PoseTemplateMatcher


class App:
    def __init__(self, config: Config) -> None:
        self._cfg     = config
        self._cameras = CameraManager(config)
        self._pose    = PoseEstimator(config)
        self._vrm     = VRMMapper(confidence=config.kp_confidence)
        self._display = DisplayManager(config)
        self._ws      = PoseWebSocketServer(config)
        self._matcher: Optional[PoseTemplateMatcher] = None
        self._running = False
        self._frame_n = 0
        # Pose-match cooldown tracking
        self._last_match_time: float = 0.0
        self._last_match_info: Optional[Tuple[str, float]] = None
        # Persisted values so the rig / debug panel stay stable between inference frames
        self._last_iso_pose:   Optional[PoseResult]   = None
        self._last_all_scores: Dict[str, float]        = {}
        # Throttle "no person detected" console spam (print at most once per 5 s)
        self._last_no_person_warn: float = 0.0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        self._cameras.open_cameras()
        if self._cfg.video_path:
            self._cameras.load_video(self._cfg.video_path)
        self._ws.start()        # Pose template matcher (runs inference on each template image at startup)
        self._matcher = PoseTemplateMatcher(
            template_dir    = self._cfg.pose_template_dir,
            estimator       = self._pose,
            match_threshold = self._cfg.pose_match_threshold,
            conf            = self._cfg.kp_confidence,
        )
    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        interval = 1.0 / max(1, self._cfg.target_fps)
        print("[App] Running — press Q to quit, 0-9 to toggle cameras.")

        while self._running:
            t0 = time.perf_counter()

            frames = self._cameras.read_frames()
            if not frames:
                time.sleep(0.01)
                continue

            all_poses:    Dict[int, List[PoseResult]] = {}
            closest_pose: Dict[int, Optional[PoseResult]] = {}

            run_inference = (self._frame_n % (self._cfg.skip_frames + 1) == 0)

            # Best single pose across all cameras (largest bbox) for rig + matching
            best_overall: Optional[PoseResult] = None
            best_cam_id:  int = 0

            for cam_idx, frame in frames.items():
                if run_inference:
                    results               = self._pose.estimate(frame)
                    all_poses[cam_idx]    = results
                    closest               = PoseEstimator.get_closest_person(results)
                    closest_pose[cam_idx] = closest

                    # Update rig pose immediately from ALL detections,
                    # before send_pose so any exception there doesn't block the rig.
                    if closest is not None:
                        if (best_overall is None
                                or closest.bbox_area > best_overall.bbox_area):
                            best_overall = closest
                            best_cam_id  = cam_idx

                    if closest is not None:
                        bones = self._vrm.map(closest)
                        self._ws.send_pose(
                            camera_id=cam_idx,
                            bones=bones,
                            keypoints=closest.keypoints.tolist(),
                        )

            # Persist best pose so rig stays visible between inference frames.
            # Also fall back to any person in all_poses in case best_overall is
            # still None (e.g. bbox_area == 0 edge-case).
            if best_overall is not None:
                self._last_iso_pose = best_overall
            elif run_inference:
                for pr_list in all_poses.values():
                    if pr_list:
                        self._last_iso_pose = pr_list[0]
                        break
                else:
                    # run_inference was True but every camera returned 0 detections.
                    now = time.perf_counter()
                    if now - self._last_no_person_warn >= 5.0:
                        print(
                            "[App] No person detected across all cameras. "
                            "Check that you are visible and well-lit in the camera frame."
                        )
                        self._last_no_person_warn = now

            # ── Pose template matching ─────────────────────────────────────────
            if run_inference and best_overall is not None and self._matcher is not None:
                match = self._matcher.match(best_overall)
                self._last_all_scores = self._matcher.last_all_scores
                now   = time.perf_counter()
                if match is not None:
                    # Enforce cooldown to avoid flooding Unity
                    if now - self._last_match_time >= self._cfg.pose_match_cooldown:
                        self._ws.send_pose_match(
                            template_name = match[0],
                            confidence    = match[1],
                            camera_id     = best_cam_id,
                        )
                        print(
                            f"[PoseMatch] {match[0]}  confidence={match[1]:.2%}  "
                            f"cam={best_cam_id}"
                        )
                        self._last_match_time = now
                    self._last_match_info = match
                else:
                    # Clear match display after cooldown period expires
                    if now - self._last_match_time > self._cfg.pose_match_cooldown:
                        self._last_match_info = None

            grid = self._display.build_grid(
                frames,
                all_poses       = all_poses,
                closest         = closest_pose,
                labels          = self._cameras.get_labels(),
                conf            = self._cfg.kp_confidence,
                iso_rig_result  = self._last_iso_pose,
                match_info      = self._last_match_info,
                all_scores      = self._last_all_scores if self._last_all_scores else None,
            )
            key = self._display.show(grid)
            self._frame_n += 1

            # ── Key handling ─────────────────────────────────────────────────
            if key == ord("q") or key == 27:        # Q or Esc
                self._running = False
            elif ord("0") <= key <= ord("9"):
                self._cameras.toggle_camera(key - ord("0"))

            # ── FPS throttle ─────────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            wait    = interval - elapsed
            if wait > 0:
                time.sleep(wait)

        self._shutdown()

    def _shutdown(self) -> None:
        self._cameras.release()
        self._display.close()
        print("[App] Stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CameraTrackingPose")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0],
                        help="Camera indices to open (default: 0)")
    parser.add_argument("--disable", type=int, nargs="*", default=[],
                        help="Camera indices to start in disabled state")
    parser.add_argument("--video",   type=str, default=None,
                        help="Path to video file (loops until stopped)")
    parser.add_argument("--device",  type=str, default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--ws-port", type=int, default=8765,
                        help="WebSocket port for Unity (default: 8765)")
    parser.add_argument("--fps",     type=int, default=30,
                        help="Target capture FPS (default: 30)")
    parser.add_argument("--skip",    type=int, default=0,
                        help="Skip N frames between inferences (0 = every frame)")
    parser.add_argument("--template-dir", type=str, default="poseTemplate",
                        help="Directory with pose template images (default: poseTemplate)")
    parser.add_argument("--match-threshold", type=float, default=0.75,
                        help="Pose match similarity threshold 0-1 (default: 0.75)")
    parser.add_argument("--match-cooldown", type=float, default=2.0,
                        help="Seconds between pose_match WebSocket events (default: 2.0)")
    parser.add_argument("--no-rig", action="store_true",
                        help="Disable the 3D isometric rig panel")
    args = parser.parse_args()

    cfg = Config(
        camera_indices      = args.cameras,
        disabled_cameras    = args.disable or [],
        video_path          = args.video,
        device              = args.device,
        ws_port             = args.ws_port,
        target_fps          = args.fps,
        skip_frames         = args.skip,
        pose_template_dir   = args.template_dir,
        pose_match_threshold= args.match_threshold,
        pose_match_cooldown = args.match_cooldown,
        show_isometric_rig  = not args.no_rig,
    )

    app = App(cfg)
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
