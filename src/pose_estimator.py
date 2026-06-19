"""
pose_estimator.py
-----------------
Wraps MMPose 1.x MMPoseInferencer for RTMPose whole-body inference.

Whole-body keypoint layout (133 total):
  0-16  : Body (COCO-17)
  17-22 : Feet  (6)
  23-90 : Face  (68)
  91-111: Left hand  (21, MediaPipe layout)
  112-132: Right hand (21, MediaPipe layout)
"""

from typing import List, Optional
import warnings

import numpy as np
from mmpose.apis import MMPoseInferencer

# torch.meshgrid indexing warning — harmless, suppress it
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

from .config import Config


class PoseResult:
    """Pose result for one detected person."""

    def __init__(
        self,
        keypoints: np.ndarray,      # (K, 2)  float32  image-space xy
        scores: np.ndarray,          # (K,)    float32
        bbox: np.ndarray,            # (5,)    float32  [x1,y1,x2,y2,score]
    ) -> None:
        self.keypoints = keypoints
        self.keypoint_scores = scores
        self.bbox = bbox

    @property
    def bbox_area(self) -> float:
        return max(0.0, (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class PoseEstimator:
    """RTMPose whole-body estimator (133 keypoints)."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._last_detection_count: int = -1   # -1 = never logged
        self._first_inference: bool = True
        print("[PoseEstimator] Loading RTMPose whole-body model…")
        self._inferencer = MMPoseInferencer(
            pose2d=config.pose2d_model,   # 'wholebody'
            device=config.device,
        )
        print("[PoseEstimator] Model ready.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """Run whole-body pose estimation on a BGR frame."""
        result_gen = self._inferencer(frame, show=False)
        raw_result = next(result_gen)

        # On first call dump the top-level keys and prediction structure so
        # mismatches between MMPose versions are immediately visible.
        if self._first_inference:
            self._first_inference = False
            top_keys = list(raw_result.keys())
            preds_raw = raw_result.get("predictions", None)
            print(f"[PoseEstimator] First-inference debug — top-level keys: {top_keys}")
            if preds_raw is not None:
                print(f"[PoseEstimator]   predictions type: {type(preds_raw).__name__}  "
                      f"len={len(preds_raw)}")
                if preds_raw and isinstance(preds_raw[0], list):
                    print(f"[PoseEstimator]   predictions[0] type: list  "
                          f"len={len(preds_raw[0])}")
                    if preds_raw[0]:
                        print(f"[PoseEstimator]   predictions[0][0] keys: "
                              f"{list(preds_raw[0][0].keys())}")
                elif preds_raw and isinstance(preds_raw[0], dict):
                    print(f"[PoseEstimator]   predictions[0] keys: "
                          f"{list(preds_raw[0].keys())}")
            else:
                print("[PoseEstimator]   WARNING — 'predictions' key missing from result!")

        # MMPose 1.x: predictions is list[list[dict]] (per-image, per-person)
        #             or list[dict] depending on input type
        preds = raw_result.get("predictions", [])
        if preds and isinstance(preds[0], list):
            preds = preds[0]           # unwrap outer image-level list

        results: List[PoseResult] = []
        for person in preds:
            kps    = np.array(person["keypoints"],       dtype=np.float32)  # (K,2)
            scores = np.array(person["keypoint_scores"], dtype=np.float32)  # (K,)
            bbox   = np.array(person["bbox"][0],         dtype=np.float32)  # (4,)
            raw_bs = person.get("bbox_score", 1.0)
            # bbox_score may be a scalar float or a 1-element list depending on MMPose version
            bscore = float(raw_bs[0]) if hasattr(raw_bs, "__len__") else float(raw_bs)
            results.append(PoseResult(kps, scores, np.append(bbox, bscore)))

        # Log when detection count changes so the user can tell if the
        # model is seeing anyone (0 → "no person detected" warning).
        n = len(results)
        if n != self._last_detection_count:
            # if n == 0:
            #     print("[PoseEstimator] WARNING — no person detected in frame. "
            #           "Check lighting, camera framing, or model confidence settings.")
            # else:
            #     print(f"[PoseEstimator] Detecting {n} person(s).")
            self._last_detection_count = n

        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_closest_person(results: List[PoseResult]) -> Optional[PoseResult]:
        """Return the person with the largest bounding box (assumed closest)."""
        return max(results, key=lambda r: r.bbox_area) if results else None
