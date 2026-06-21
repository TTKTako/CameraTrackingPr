"""
pose_estimator.py
-----------------
True 3D Pose Estimator using MediaPipe Holistic.
Provides both 2D pixel coordinates (for video overlay) and True 3D world coordinates (for Unity).
"""

from typing import List, Optional
import numpy as np
import cv2

import mediapipe as mp
import mediapipe.solutions.holistic as mp_holistic # Explicit, safe import

from .config import Config

class PoseResult:
    def __init__(
        self,
        keypoints: np.ndarray,      # (133, 2) 2D Pixels for UI
        keypoints_3d: np.ndarray,   # (133, 3) True 3D Meters for Unity/VRM
        scores: np.ndarray,         # (133,)   Confidence
        bbox: np.ndarray,           # (5,)     [x1, y1, x2, y2, score]
    ) -> None:
        self.keypoints = keypoints
        self.keypoints_3d = keypoints_3d
        self.keypoint_scores = scores
        self.bbox = bbox

    @property
    def bbox_area(self) -> float:
        return max(0.0, (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))

class PoseEstimator:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        print("[PoseEstimator] Loading MediaPipe Holistic 3D...")
        
        # --- USE THE EXPLICIT IMPORT DIRECTLY ---
        self.mp_holistic = mp_holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Map MediaPipe Pose indices to our COCO-17 format
        self.mp_to_coco = {
            0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 
            7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 
            13: 25, 14: 26, 15: 27, 16: 28
        }
        print("[PoseEstimator] 3D Model ready.")

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        # MediaPipe requires RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        # If no body is detected, return empty list
        if not results.pose_landmarks:
            return []

        h, w, _ = frame.shape
        
        # Initialize arrays for our 133 format
        kps_2d = np.zeros((133, 2), dtype=np.float32)
        kps_3d = np.zeros((133, 3), dtype=np.float32)
        scores = np.zeros((133,), dtype=np.float32)

        # 1. Process Body (True 3D World Landmarks)
        if results.pose_world_landmarks and results.pose_landmarks:
            world_lms = results.pose_world_landmarks.landmark
            pixel_lms = results.pose_landmarks.landmark
            
            for coco_idx, mp_idx in self.mp_to_coco.items():
                pixel_lm = pixel_lms[mp_idx]
                world_lm = world_lms[mp_idx]
                
                # 2D for Drawing & Template Matching
                kps_2d[coco_idx] = [pixel_lm.x * w, pixel_lm.y * h]
                # True 3D for Unity VRM & Isometric Plot (x, y, z in meters)
                kps_3d[coco_idx] = [world_lm.x, world_lm.y, world_lm.z]
                scores[coco_idx] = pixel_lm.visibility

        # Calculate bounding box from 2D points
        valid_pts = kps_2d[scores > 0.1]
        if len(valid_pts) > 0:
            x_min, y_min = np.min(valid_pts, axis=0)
            x_max, y_max = np.max(valid_pts, axis=0)
            # Add padding
            bbox = np.array([x_min-20, y_min-20, x_max+20, y_max+20, 1.0])
        else:
            bbox = np.array([0, 0, 0, 0, 0])

        return [PoseResult(kps_2d, kps_3d, scores, bbox)]

    @staticmethod
    def get_closest_person(results: List[PoseResult]) -> Optional[PoseResult]:
        return max(results, key=lambda r: r.bbox_area) if results else None