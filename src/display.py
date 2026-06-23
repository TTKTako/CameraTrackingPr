"""
display.py
----------
Thread-safe Matplotlib 3D visualization. Renders the 3D rig into an image buffer
with dual angles (Front and Side) and stitches it with the camera feed.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Forces thread-safe, off-screen rendering
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .pose_estimator import PoseResult

class DisplayManager:
    def __init__(self, config):
        self._cfg = config
        
        # Setup Matplotlib for off-screen rendering (Wider figure for two plots)
        self.fig = plt.figure(figsize=(10, 5), dpi=100)
        
        # --- PLOT 1: Front View ---
        self.ax_front = self.fig.add_subplot(121, projection='3d')
        self.ax_front.set_title("Front View")
        self.ax_front.set_xlim([-400, 400])
        self.ax_front.set_ylim([-400, 400])
        self.ax_front.set_zlim([-400, 400])
        self.ax_front.view_init(elev=15, azim=-90) # Looking straight on
        
        # --- PLOT 2: Side View ---
        self.ax_side = self.fig.add_subplot(122, projection='3d')
        self.ax_side.set_title("Side View")
        self.ax_side.set_xlim([-400, 400])
        self.ax_side.set_ylim([-400, 400])
        self.ax_side.set_zlim([-400, 400])
        self.ax_side.view_init(elev=15, azim=0) # Looking from the side
        
        self.fig.tight_layout()

        # Create empty 3D line objects for BOTH views
        self.lines_front = {
            'arms': self.ax_front.plot([], [], [], 'r-', lw=2, marker='o')[0],
            'legs': self.ax_front.plot([], [], [], 'b-', lw=2, marker='o')[0],
            'spine': self.ax_front.plot([], [], [], 'g-', lw=3, marker='o')[0]
        }
        self.lines_side = {
            'arms': self.ax_side.plot([], [], [], 'r-', lw=2, marker='o')[0],
            'legs': self.ax_side.plot([], [], [], 'b-', lw=2, marker='o')[0],
            'spine': self.ax_side.plot([], [], [], 'g-', lw=3, marker='o')[0]
        }

    def build_grid(self, frames: Dict[int, np.ndarray], closest: Dict[int, Optional[PoseResult]], **kwargs):
        if not frames: return None  # Check if the dictionary is empty instead
        
        # Dynamically get the first active camera index
        cam_idx = next(iter(frames.keys()))
        
        frame = frames[cam_idx].copy()
        pose = closest.get(cam_idx)

        # 1. Update 3D data if a person is found
        if pose is not None:
            x1, y1, x2, y2, _ = pose.bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            self._update_3d_plot(pose)
        else:
            # Hide lines on both plots if no person is detected
            for lines in [self.lines_front, self.lines_side]:
                for line in lines.values():
                    line.set_data_3d([], [], [])
        
        # 2. Render the Matplotlib plots to a NumPy Image Array
        self.fig.canvas.draw()
        rgba_buffer = np.asarray(self.fig.canvas.buffer_rgba())
        plot_img = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
        
        # 3. Resize the Camera Frame to match the Plot height
        target_height = plot_img.shape[0]
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_width = int(target_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (target_width, target_height))
        
        # 4. Stitch them side-by-side (Camera | Front | Side)
        combined_grid = cv2.hconcat([frame_resized, plot_img])
        
        return combined_grid

    def show(self, grid: np.ndarray) -> int:
        if grid is not None:
            cv2.imshow("Camera & 3D Rig", grid)
        return cv2.waitKey(1) & 0xFF

    def _update_3d_plot(self, pose: PoseResult):
        # --- NEW: Use True 3D Data ---
        # Get real X, Y, Z coordinates from MediaPipe in meters
        kps_3d = pose.keypoints_3d
        
        # Scale it up slightly so it fits the [-400, 400] Matplotlib grid
        scale_factor = 500.0 
        x = kps_3d[:, 0] * scale_factor
        y = -kps_3d[:, 1] * scale_factor
        z = -kps_3d[:, 2] * scale_factor # TRUE DEPTH!

        # Auto-Center the rig
        center_x = (x[11] + x[12]) / 2 
        center_y = (y[11] + y[12]) / 2
        center_z = (z[11] + z[12]) / 2

        x = x - center_x
        y = y - center_y
        z = z - center_z

        hip_cx, hip_cy, hip_cz = (x[11]+x[12])/2, (y[11]+y[12])/2, (z[11]+z[12])/2
        sh_cx, sh_cy, sh_cz = (x[5]+x[6])/2, (y[5]+y[6])/2, (z[5]+z[6])/2

        for lines in [self.lines_front, self.lines_side]:
            # Inject True Z depth into the set_data_3d functions
            lines['arms'].set_data_3d(x[[10, 8, 6, 5, 7, 9]], z[[10, 8, 6, 5, 7, 9]], y[[10, 8, 6, 5, 7, 9]])
            lines['legs'].set_data_3d(x[[16, 14, 12, 11, 13, 15]], z[[16, 14, 12, 11, 13, 15]], y[[16, 14, 12, 11, 13, 15]])
            lines['spine'].set_data_3d([hip_cx, sh_cx, x[0]], [hip_cz, sh_cz, z[0]], [hip_cy, sh_cy, y[0]])
            
    def close(self):
        plt.close(self.fig)
        cv2.destroyAllWindows()