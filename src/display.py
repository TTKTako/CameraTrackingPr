"""
display.py
----------
OpenCV-based debug/preview display.

Features:
  • Draws COCO-17 skeleton + keypoints on each camera frame
  • Highlights the "closest person" (largest bbox) with a green overlay
  • Auto-builds an N-column grid from all active camera frames
  • Labels each cell with the camera index
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .pose_estimator import PoseResult

# ── Isometric projection constants ────────────────────────────────────────────
# Standard isometric (dimetric) view from upper-right:
#   iso_x = world_x * cos(30°)               (world_z = 0)
#   iso_y = world_x * sin(30°) − world_y     (world_z = 0)
_ISO_COS = math.cos(math.radians(30))   # ≈ 0.866
_ISO_SIN = math.sin(math.radians(30))   # ≈ 0.500

# ── COCO-17 skeleton connections (joint pairs) ────────────────────────────────
_SKELETON = [
    (0, 1), (0, 2),               # nose → eyes
    (1, 3), (2, 4),               # eyes → ears
    (5, 6),                       # shoulder bar
    (5, 7), (7, 9),               # left arm
    (6, 8), (8, 10),              # right arm
    (5, 11), (6, 12),             # torso sides
    (11, 12),                     # hip bar
    (11, 13), (13, 15),           # left leg
    (12, 14), (14, 16),           # right leg
]

_LIMB_COLORS = [
    (255, 200, 100), (255, 200, 100),          # head
    (255, 200, 100), (255, 200, 100),
    (200, 255, 100),                           # shoulders
    (100, 200, 255), (100, 200, 255),          # left arm
    (255, 100, 200), (255, 100, 200),          # right arm
    (200, 255, 100), (200, 255, 100),          # torso
    (200, 255, 100),
    (100, 255, 200), (100, 255, 200),          # left leg
    (200, 100, 255), (200, 100, 255),          # right leg
]


class DisplayManager:
    def __init__(self, config: Config) -> None:
        self._cfg = config

    # ── Skeleton drawing ──────────────────────────────────────────────────────

    def draw_pose(
        self,
        frame: np.ndarray,
        result: PoseResult,
        highlight: bool = False,
        conf: float = 0.3,
    ) -> np.ndarray:
        """Draw bounding box + skeleton on a copy of frame."""
        frame = frame.copy()
        kps    = result.keypoints
        scores = result.keypoint_scores

        # Bounding box
        x1, y1, x2, y2 = (int(v) for v in result.bbox[:4])
        box_color = (0, 255, 80) if highlight else (80, 200, 80)
        thickness = 3 if highlight else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Limbs
        for i, (j1, j2) in enumerate(_SKELETON):
            if j1 >= len(kps) or j2 >= len(kps):
                continue
            if scores[j1] < conf or scores[j2] < conf:
                continue
            pt1 = (int(kps[j1][0]), int(kps[j1][1]))
            pt2 = (int(kps[j2][0]), int(kps[j2][1]))
            color = _LIMB_COLORS[i] if i < len(_LIMB_COLORS) else (255, 255, 255)
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # Keypoints (body only, 0-16)
        for j in range(min(17, len(kps))):
            if scores[j] < conf:
                continue
            pt = (int(kps[j][0]), int(kps[j][1]))
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, (0, 0, 0), 1)

        return frame

    # ── Isometric 3-D rig panel ───────────────────────────────────────────────

    def draw_isometric_rig(
        self,
        result: Optional[PoseResult],
        conf: float = 0.3,
        match_info: Optional[Tuple[str, float]] = None,
    ) -> np.ndarray:
        """
        Render the COCO-17 skeleton as a 3-D isometric stick figure.

        The pose is projected with the standard isometric formula so that
        depth (left/right displacement) creates vertical parallax, giving
        a top-front-right camera perspective without true depth data.
        """
        size  = self._cfg.iso_rig_size
        panel = np.full((size, size, 3), (15, 20, 35), dtype=np.uint8)

        # ── Placeholder when no pose available ─────────────────────────────
        if result is None:
            cv2.putText(panel, "3D Rig (Isometric)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 220), 1, cv2.LINE_AA)
            # Draw faint grid to hint at 3-D space even without a pose
            mid = size // 2
            for gx in np.arange(-1.5, 1.6, 0.4):
                px = int(mid + float(gx) * _ISO_COS * size * 0.3)
                cv2.line(panel, (px, mid - 60), (px, mid + 80), (30, 36, 52), 1)
            cv2.putText(panel, "Waiting for pose...",
                        (size // 2 - 80, size // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 70, 100), 1, cv2.LINE_AA)
            return panel

        kps    = result.keypoints[:17]
        scores = result.keypoint_scores[:17]
        # Very permissive threshold for the rig — we want to draw whatever joints
        # the detector found, even at low confidence.  The stricter `conf` still
        # applies to the camera-overlay skeleton drawn by draw_pose().
        rig_conf = 0.05
        valid    = scores >= rig_conf

        if not np.any(valid):
            cv2.putText(panel, "3D Rig (Isometric)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 220), 1, cv2.LINE_AA)
            cv2.putText(panel, "No joints visible", (10, size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 90), 1, cv2.LINE_AA)
            return panel

        # ── Normalise to [-1, 1] centred on the detected bounding box ──────
        pts   = kps[valid]
        cx    = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        cy    = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
        span  = max(
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            1.0,
        ) / 1.7
        norm  = (kps - np.array([cx, cy])) / span   # (17, 2)

        # ── Isometric projection helper ────────────────────────────────────
        half  = size // 2
        scale = size * 0.38

        def project(j: int) -> Optional[Tuple[int, int]]:
            if not valid[j]:
                return None
            nx, ny     = float(norm[j, 0]), float(norm[j, 1])
            world_y    = -ny          # flip: screen-down → world-up
            iso_x      = nx * _ISO_COS
            iso_y      = nx * _ISO_SIN - world_y   # = nx*sin30 + ny
            px         = int(half + iso_x * scale)
            py         = int(half + iso_y * scale)
            return (px, py)

        # ── Floor grid (two crossed sets of diagonal lines) ────────────────
        # Compute approximate ankle iso-y for floor positioning
        ankle_iso_ys = []
        for j in [15, 16]:
            if valid[j]:
                nx, ny  = float(norm[j, 0]), float(norm[j, 1])
                ankle_iso_ys.append(nx * _ISO_SIN + ny)
        floor_iso_y = max(ankle_iso_ys) + 0.12 if ankle_iso_ys else 0.9
        floor_py    = int(half + floor_iso_y * scale)

        grid_col = (40, 48, 68)
        for gx in np.arange(-1.5, 1.6, 0.35):
            gx_f   = float(gx)
            px_top = int(half + gx_f * _ISO_COS * scale)
            py_top = int(half + (gx_f * _ISO_SIN - 0.15) * scale)
            px_bot = int(half + gx_f * _ISO_COS * scale)
            py_bot = floor_py + 10
            # vertical iso-lines
            cv2.line(panel, (px_top, py_top), (px_bot, py_bot), grid_col, 1, cv2.LINE_AA)
        for gz in np.arange(-1.5, 1.6, 0.35):
            gz_f   = float(gz)
            px_l   = int(half + (-1.5 - gz_f) * _ISO_COS * scale)
            py_l   = int(floor_py)
            px_r   = int(half + ( 1.5 - gz_f) * _ISO_COS * scale)
            py_r   = int(floor_py)
            cv2.line(panel, (px_l, floor_py), (px_r, floor_py), grid_col, 1, cv2.LINE_AA)

        # Shadow dots at ankle positions
        for j in [15, 16]:
            pt = project(j)
            if pt:
                shadow_y = floor_py + 4
                cv2.ellipse(panel, (pt[0], shadow_y), (12, 4),
                            0, 0, 360, (30, 35, 50), -1)

        # ── Limbs ──────────────────────────────────────────────────────────
        for i, (j1, j2) in enumerate(_SKELETON):
            if j1 >= 17 or j2 >= 17:
                continue
            pt1, pt2 = project(j1), project(j2)
            if pt1 is None or pt2 is None:
                continue
            color = _LIMB_COLORS[i] if i < len(_LIMB_COLORS) else (200, 200, 200)
            cv2.line(panel, pt1, pt2, color, 3, cv2.LINE_AA)

        # ── Joints ─────────────────────────────────────────────────────────
        for j in range(17):
            pt = project(j)
            if pt is None:
                continue
            r = 7 if j in (0, 5, 6, 11, 12) else 5   # larger for torso anchors
            cv2.circle(panel, pt, r, (240, 240, 255), -1)
            cv2.circle(panel, pt, r, (0,   0,   0),   1)

        # ── Axis indicator (small) ─────────────────────────────────────────
        ax, ay = 30, size - 30
        arrow_len = 18
        # X axis (right, red)
        cv2.arrowedLine(panel, (ax, ay),
                        (ax + arrow_len, ay), (80, 80, 220), 1, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(panel, "X", (ax + arrow_len + 2, ay + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 220), 1)
        # Y axis (up, green)
        cv2.arrowedLine(panel, (ax, ay),
                        (ax, ay - arrow_len), (80, 200, 80), 1, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(panel, "Y", (ax - 10, ay - arrow_len - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 200, 80), 1)

        # ── Labels ─────────────────────────────────────────────────────────
        cv2.putText(panel, "3D Rig (Isometric)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 220), 1, cv2.LINE_AA)

        if match_info:
            name, score = match_info
            label = f"MATCH: {name}  {score:.0%}"
            # Pulsing green background bar
            bar_y = size - 32
            cv2.rectangle(panel, (0, bar_y - 4), (size, size), (0, 60, 30), -1)
            cv2.putText(panel, label, (8, size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

        return panel

    # ── Pose match debug overlay (drawn on a camera cell) ─────────────────────

    def draw_match_debug_overlay(
        self,
        cell: np.ndarray,
        all_scores: Dict[str, float],
        match_info: Optional[Tuple[str, float]],
        threshold: float = 0.82,
    ) -> np.ndarray:
        """
        Draw a semi-transparent debug panel on the bottom-left of a camera cell
        showing per-template similarity scores as horizontal bars.
        """
        if not all_scores:
            return cell

        cell = cell.copy()
        h, w  = cell.shape[:2]
        n     = len(all_scores)
        row_h = 18
        pad   = 6
        bar_max_w = 130
        panel_h   = pad + n * row_h + pad
        panel_w   = bar_max_w + 120
        y0        = h - panel_h - 4
        x0        = 4

        # Semi-transparent background
        overlay = cell.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (10, 12, 20), -1)
        cv2.addWeighted(overlay, 0.65, cell, 0.35, 0, cell)

        # Per-template bars
        matched_name = match_info[0] if match_info else ""
        for i, (name, score) in enumerate(sorted(all_scores.items())):
            ry = y0 + pad + i * row_h
            bar_w = int(score * bar_max_w)
            is_match = (name == matched_name)

            # Background bar track
            cv2.rectangle(cell,
                          (x0 + 4, ry + 2),
                          (x0 + 4 + bar_max_w, ry + row_h - 2),
                          (30, 35, 50), -1)
            # Filled bar — green if matched, yellow/red based on score
            if is_match:
                bar_color = (0, 220, 100)
            elif score >= threshold * 0.9:
                bar_color = (0, 180, 255)
            else:
                bar_color = (60, 100, 160)
            cv2.rectangle(cell,
                          (x0 + 4, ry + 2),
                          (x0 + 4 + bar_w, ry + row_h - 2),
                          bar_color, -1)

            # Threshold tick mark
            thresh_x = x0 + 4 + int(threshold * bar_max_w)
            cv2.line(cell, (thresh_x, ry + 1), (thresh_x, ry + row_h - 1),
                     (200, 200, 50), 1)

            # Label
            short_name = name[:14] + ".." if len(name) > 14 else name
            label_color = (0, 255, 120) if is_match else (180, 180, 180)
            cv2.putText(cell, f"{short_name} {score:.0%}",
                        (x0 + bar_max_w + 8, ry + row_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1, cv2.LINE_AA)

        # Header
        header = "POSE MATCH" if match_info else "pose scores"
        hcolor = (0, 255, 120) if match_info else (140, 140, 140)
        cv2.putText(cell, header, (x0 + 4, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, hcolor, 1, cv2.LINE_AA)

        # Big match banner at top if matched
        if match_info:
            name, score = match_info
            banner = f">> {name}  {score:.0%} <<"
            bw, bh = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
            bx = (w - bw) // 2
            by = 52
            cv2.rectangle(cell, (bx - 6, by - 22), (bx + bw + 6, by + 6),
                          (0, 50, 25), -1)
            cv2.putText(cell, banner, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2, cv2.LINE_AA)

        return cell

    # ── Grid builder ─────────────────────────────────────────────────────────

    def build_grid(
        self,
        frames: Dict[int, np.ndarray],
        all_poses: Optional[Dict[int, List[PoseResult]]] = None,
        closest:   Optional[Dict[int, Optional[PoseResult]]] = None,
        labels:    Optional[Dict[int, str]] = None,
        conf: float = 0.3,
        iso_rig_result: Optional[PoseResult] = None,
        match_info: Optional[Tuple[str, float]] = None,
        all_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Build an auto-layout grid image.

        Parameters
        ----------
        frames         : {cam_idx: BGR frame}
        all_poses      : {cam_idx: [PoseResult, ...]}   – ALL detected people
        closest        : {cam_idx: PoseResult | None}   – highlighted closest
        labels         : {cam_idx: custom label text}
        iso_rig_result : pose for the 3-D rig panel (None = placeholder)
        match_info     : (template_name, score) best match above threshold
        all_scores     : {template_name: score} all per-template scores for debug
        """
        if not frames:
            return np.zeros(
                (self._cfg.display_height, self._cfg.display_width, 3), np.uint8
            )

        show_rig = self._cfg.show_isometric_rig
        n_cams   = len(frames)
        n_cells  = n_cams + (1 if show_rig else 0)
        cols     = max(1, math.ceil(math.sqrt(n_cells)))
        rows     = max(1, math.ceil(n_cells / cols))
        cell_w   = self._cfg.display_width  // cols
        cell_h   = self._cfg.display_height // rows

        # Camera index with the tracked person (for debug overlay)
        best_cam = None
        if closest:
            for cam_idx, pr in closest.items():
                if pr is not None:
                    if best_cam is None or (
                        closest.get(best_cam) is not None
                        and pr.bbox_area > closest[best_cam].bbox_area
                    ):
                        best_cam = cam_idx

        cells = []
        for cam_idx in sorted(frames.keys()):
            cell = frames[cam_idx].copy()

            # Draw all detected people
            if all_poses and cam_idx in all_poses:
                for person in all_poses[cam_idx]:
                    is_closest = (
                        closest
                        and cam_idx in closest
                        and person is closest[cam_idx]
                    )
                    cell = self.draw_pose(cell, person, highlight=is_closest, conf=conf)

            # Pose match debug overlay — only on the camera with the tracked person
            if cam_idx == best_cam and all_scores is not None:
                cell = self.draw_match_debug_overlay(
                    cell, all_scores, match_info,
                    threshold=self._cfg.pose_match_threshold,
                )

            # Camera label
            lbl = (labels or {}).get(cam_idx, f"Cam {cam_idx}")
            cv2.putText(cell, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cells.append(cv2.resize(cell, (cell_w, cell_h)))

        # Always append isometric rig panel (placeholder when no pose)
        if show_rig:
            rig = self.draw_isometric_rig(iso_rig_result, conf=conf, match_info=match_info)
            cells.append(cv2.resize(rig, (cell_w, cell_h)))

        # Pad grid to rows × cols
        blank = np.zeros((cell_h, cell_w, 3), np.uint8)
        while len(cells) < rows * cols:
            cells.append(blank)

        row_imgs = [np.hstack(cells[r * cols: (r + 1) * cols]) for r in range(rows)]
        return np.vstack(row_imgs)

    # ── Window helpers ────────────────────────────────────────────────────────

    def show(self, grid: np.ndarray) -> int:
        """Display grid. Returns the key pressed (-1 if none)."""
        cv2.imshow(self._cfg.window_name, grid)
        return cv2.waitKey(1)

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()
