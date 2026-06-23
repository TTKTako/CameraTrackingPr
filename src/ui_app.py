"""
ui_app.py
---------
Tkinter GUI for CameraTrackingPose.

Layout
──────
┌──────────────────────────────────────────────────────────────────┐
│  Header:  CameraTrackingPose          ● Idle  [▶ Start] [■ Stop] │
├──────────────────────────────┬───────────────────────────────────┤
│                              │  ┌─ Cameras ─────────────────┐   │
│   Camera feed                │  │ [✓] Dev[0▼] ● Label    [✕]│   │
│   (embedded, auto-grid)      │  │ [✓] Dev[1▼] ○ Label    [✕]│   │
│                              │  │  [+ Add Camera] [⟳ Scan]  │   │
│                              │  └───────────────────────────┘   │
│                              │  ┌─ Video File ───────────────┐   │
│                              │  │  [path/to/file.mp4       ] │   │
│                              │  │  [Browse…] [Clear]         │   │
│                              │  │  [✓] Loop video            │   │
│                              │  └───────────────────────────┘   │
│                              │  ┌─ Settings ─────────────────┐   │
│                              │  │  Device  [cuda:0 ▼]        │   │
│                              │  │  FPS     [30]              │   │
│                              │  │  Skip    [0]               │   │
│                              │  │  Conf    [──●──]  0.30     │   │
│                              │  └───────────────────────────┘   │
│                              │  ┌─ WebSocket → Unity ────────┐   │
│                              │  │  Port [8765]               │   │
│                              │  │  ● Waiting…   Clients: 0   │   │
│                              │  └───────────────────────────┘   │
│                              │  ┌─ Live Stats ───────────────┐   │
│                              │  │  FPS: 28.4                 │   │
│                              │  │  People detected: 2        │   │
│                              │  └───────────────────────────┘   │
└──────────────────────────────┴───────────────────────────────────┘

Thread model
────────────
  Main thread  : tkinter event loop + canvas rendering
  Pipeline thread (daemon) : camera capture → RTMPose → VRM → WebSocket
  Communication : queue.Queue (frames)  +  direct method calls (camera toggles)
"""

import os
import queue
import threading
import time
import copy
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import Config
from .camera_manager import CameraManager, scan_cameras
from .display import DisplayManager
from .pose_estimator import PoseEstimator, PoseResult
from .vrm_mapper import VRMMapper
from .websocket_server import PoseWebSocketServer


# ── Data passed from pipeline → UI ────────────────────────────────────────────

@dataclass
class _FrameUpdate:
    grid: np.ndarray
    fps: float
    people_counts: Dict[int, int]
    ws_clients: int


# ── Pipeline worker thread ─────────────────────────────────────────────────────

class _PipelineThread(threading.Thread):
    """
    Runs: camera capture → RTMPose inference → VRM mapping → WebSocket broadcast.
    Puts _FrameUpdate objects into out_queue for the UI to render.
    """

    def __init__(self, config: Config, out_queue: "queue.Queue[_FrameUpdate]",
                 stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="pipeline")
        self._cfg        = config
        self._out_queue  = out_queue
        self._stop_event = stop_event
        self.error: Optional[str] = None

        # Created inside run() — keep on this thread
        self._cam_mgr:   Optional[CameraManager]    = None
        self._estimator: Optional[PoseEstimator]    = None
        self._vrm:       Optional[VRMMapper]        = None
        self._display:   Optional[DisplayManager]   = None
        self._ws:        Optional[PoseWebSocketServer] = None

    # ── Thread entry ──────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._setup()
            self._loop()
        except Exception as exc:
            import traceback
            self.error = traceback.format_exc()
        finally:
            self._cleanup()

    def _setup(self) -> None:
        self._cam_mgr  = CameraManager(self._cfg)
        self._cam_mgr.open_cameras()
        if self._cfg.video_path:
            self._cam_mgr.load_video(self._cfg.video_path)
        self._estimator = PoseEstimator(self._cfg)
        self._vrm       = VRMMapper(self._cfg.kp_confidence)
        self._display   = DisplayManager(self._cfg)
        self._ws        = PoseWebSocketServer(self._cfg)
        self._ws.start()

    def _loop(self) -> None:
        interval  = 1.0 / max(1, self._cfg.target_fps)
        frame_n   = 0
        last_t    = time.perf_counter()
        fps_acc   = 0.0
        fps_cnt   = 0
        fps       = 0.0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            frames = self._cam_mgr.read_frames()
            if not frames:
                time.sleep(0.01)
                continue

            all_poses:     Dict[int, List[PoseResult]] = {}
            closest_pose:  Dict[int, Optional[PoseResult]] = {}
            people_counts: Dict[int, int] = {}
            run_inf = (frame_n % (self._cfg.skip_frames + 1) == 0)

            # 1. PROCESS ALL CAMERAS
            for cam_idx, frame in frames.items():
                if run_inf:
                    results              = self._estimator.estimate(frame)
                    all_poses[cam_idx]   = results
                    closest              = PoseEstimator.get_closest_person(results)
                    closest_pose[cam_idx]= closest
                    people_counts[cam_idx] = len(results)
                    # Notice: We REMOVED the self._ws.send_pose from inside this loop!

            # 2. MULTI-CAMERA AI FUSION!
            if run_inf:
                valid_poses = [p for p in closest_pose.values() if p is not None]
                
                if valid_poses:
                    # Start with a copy of the first camera's pose
                    fused_pose = copy.deepcopy(valid_poses[0])
                    
                    # If we have multiple cameras, merge them mathematically!
                    if len(valid_poses) > 1:
                        num_kps = len(fused_pose.keypoints_3d)
                        merged_3d = np.zeros_like(fused_pose.keypoints_3d)
                        merged_scores = np.zeros_like(fused_pose.keypoint_scores)
                        weight_sum = np.zeros(num_kps)
                        
                        for p in valid_poses:
                            for i in range(num_kps):
                                score = p.keypoint_scores[i]
                                # Squaring the score heavily punishes bad angles/occlusions
                                weight = score ** 2 
                                
                                merged_3d[i] += p.keypoints_3d[i] * weight
                                weight_sum[i] += weight
                                # Keep the highest confidence score for the UI
                                merged_scores[i] = max(merged_scores[i], score) 
                                
                        for i in range(num_kps):
                            if weight_sum[i] > 0:
                                merged_3d[i] /= weight_sum[i]
                                
                        fused_pose.keypoints_3d = merged_3d
                        fused_pose.keypoint_scores = merged_scores

                    # 3. Map to VRM and Send exactly ONCE per frame
                    bones = self._vrm.map(fused_pose)
                    self._ws.send_pose(
                        camera_id=99, # Indicates a merged data packet
                        bones=bones,
                        keypoints=fused_pose.keypoints.tolist(),
                    )

            # --- REMAINDER OF YOUR LOOP STAYS THE SAME ---
            grid = self._display.build_grid(
                frames,
                all_poses=all_poses,
                closest=closest_pose,
                labels=self._cam_mgr.get_labels(),
                conf=self._cfg.kp_confidence,
            )

            # Rolling FPS
            frame_n += 1
            now      = time.perf_counter()
            fps_acc += 1.0 / max(1e-9, now - last_t)
            fps_cnt += 1
            last_t   = now
            if fps_cnt >= 10:
                fps     = fps_acc / fps_cnt
                fps_acc = fps_cnt = 0

            update = _FrameUpdate(
                grid=grid,
                fps=fps,
                people_counts=people_counts,
                ws_clients=self._ws.connected_clients,
            )
            try:
                self._out_queue.put_nowait(update)
            except queue.Full:
                pass  # drop frame — UI is behind

            elapsed = time.perf_counter() - t0
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    def _cleanup(self) -> None:
        if self._cam_mgr:
            self._cam_mgr.release()

    # ── Thread-safe camera controls (called from UI thread) ──────────────────

    def enable_camera(self, index: int) -> None:
        if self._cam_mgr:
            self._cam_mgr.enable_camera(index)

    def disable_camera(self, index: int) -> None:
        if self._cam_mgr:
            self._cam_mgr.disable_camera(index)

    def send_manual_match(self, template_name: str, confidence: float) -> None:
        """Thread-safe: fire a pose_match WebSocket event from the UI."""
        if self._ws:
            self._ws.send_pose_match(
                template_name=template_name,
                confidence=confidence,
                camera_id=0,
            )


# ── Main application window ───────────────────────────────────────────────────

class CameraTrackingApp(tk.Tk):

    _FEED_W  = 800
    _FEED_H  = 540
    _PANEL_W = 290

    def __init__(self) -> None:
        super().__init__()
        self.title("CameraTrackingPose")
        self.configure(bg="#1e1e2e")
        self.minsize(self._FEED_W + self._PANEL_W + 24, 620)

        style = ttk.Style(self)
        style.theme_use("clam")
        self._apply_theme(style)

        # Runtime state
        self._pipeline:    Optional[_PipelineThread] = None
        self._stop_event   = threading.Event()
        self._frame_queue: "queue.Queue[_FrameUpdate]" = queue.Queue(maxsize=2)
        self._photo:       Optional[ImageTk.PhotoImage] = None

        # Camera entries: list of dict {index, device_var, enabled_var, label_var, status_lbl, row}
        self._camera_entries: List[Dict] = []
        # Available devices discovered by scan
        self._available_devices: List[Tuple[int, str]] = [(0, "0 — (not scanned)")]

        self._build_ui()
        self._add_camera_entry(0)   # default camera 0

        self.after(33, self._ui_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Dark theme ────────────────────────────────────────────────────────────

    def _apply_theme(self, s: ttk.Style) -> None:
        BG, FG      = "#1e1e2e", "#cdd6f4"
        ACC         = "#89b4fa"
        BTN, BTN_H  = "#313244", "#45475a"
        ENT         = "#313244"

        s.configure(".",                background=BG, foreground=FG, font=("Segoe UI", 9))
        s.configure("TFrame",           background=BG)
        s.configure("TLabel",           background=BG, foreground=FG)
        s.configure("TLabelframe",      background=BG, foreground=ACC, relief="groove")
        s.configure("TLabelframe.Label",background=BG, foreground=ACC, font=("Segoe UI", 9, "bold"))
        s.configure("TButton",          background=BTN, foreground=FG, relief="flat", padding=(8, 4))
        s.map("TButton",                background=[("active", BTN_H)])
        s.configure("TCheckbutton",     background=BG, foreground=FG)
        s.map("TCheckbutton",           background=[("active", BG)])
        s.configure("TEntry",           fieldbackground=ENT, foreground=FG, insertcolor=FG)
        s.configure("TCombobox",        fieldbackground=ENT, foreground=FG, selectbackground=ENT)
        s.configure("TSpinbox",         fieldbackground=ENT, foreground=FG, insertcolor=FG)
        s.configure("TScale",           background=BG, troughcolor=BTN)
        s.configure("TScrollbar",       background=BTN, troughcolor=BG, arrowcolor=FG)

        s.configure("Start.TButton", background="#a6e3a1", foreground="#1e1e2e",
                    font=("Segoe UI", 10, "bold"), padding=(12, 5))
        s.map("Start.TButton",       background=[("active", "#94d488"), ("disabled", BTN)])

        s.configure("Stop.TButton",  background="#f38ba8", foreground="#1e1e2e",
                    font=("Segoe UI", 10, "bold"), padding=(12, 5))
        s.map("Stop.TButton",        background=[("active", "#e07a95"), ("disabled", BTN)])

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Header bar
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(hdr, text="CameraTrackingPose",
                  font=("Segoe UI", 13, "bold"),
                  foreground="#89b4fa").pack(side="left")

        self._btn_stop  = ttk.Button(hdr, text="■  Stop",  style="Stop.TButton",
                                     command=self._on_stop, state="disabled")
        self._btn_stop.pack(side="right", padx=(4, 0))

        self._btn_start = ttk.Button(hdr, text="▶  Start", style="Start.TButton",
                                     command=self._on_start)
        self._btn_start.pack(side="right", padx=4)

        self._lbl_status = ttk.Label(hdr, text="● Idle",
                                     foreground="#a6e3a1", font=("Segoe UI", 9))
        self._lbl_status.pack(side="right", padx=14)

        ttk.Separator(self).pack(fill="x", padx=8, pady=2)

        # Main area
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Left — live feed canvas
        self._canvas = tk.Canvas(main, width=self._FEED_W, height=self._FEED_H,
                                 bg="#11111b", highlightthickness=1,
                                 highlightbackground="#313244")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._canvas.create_text(self._FEED_W // 2, self._FEED_H // 2,
                                 text="Press  ▶ Start  to begin",
                                 fill="#585b70", font=("Segoe UI", 14),
                                 tags="placeholder")

        # Right — scrollable control panel
        panel = ttk.Frame(main, width=self._PANEL_W)
        panel.pack(side="right", fill="y", padx=(10, 0))
        panel.pack_propagate(False)

        pane = self._make_scrollable(panel)
        self._build_cameras_section(pane)
        self._build_video_section(pane)
        self._build_settings_section(pane)
        self._build_ws_section(pane)
        self._build_manual_trigger_section(pane)
        self._build_stats_section(pane)

    def _make_scrollable(self, parent: ttk.Frame) -> ttk.Frame:
        cv   = tk.Canvas(parent, bg="#1e1e2e", highlightthickness=0)
        sb   = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv)
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        return inner

    # ── Cameras section ───────────────────────────────────────────────────────

    def _build_cameras_section(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" Cameras ", padding=6)
        lf.pack(fill="x", padx=4, pady=(4, 6))

        self._cameras_container = ttk.Frame(lf)
        self._cameras_container.pack(fill="x")

        btn_row = ttk.Frame(lf)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="+ Add Camera",
                   command=self._on_add_camera).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="⟳ Scan Devices",
                   command=self._on_scan_cameras).pack(side="left")
        self._scan_lbl = ttk.Label(btn_row, text="", foreground="#7f849c",
                                   font=("Segoe UI", 8))
        self._scan_lbl.pack(side="left", padx=(6, 0))

    def _device_values(self) -> List[str]:
        """Return combobox values from the current discovered device list."""
        return [desc for _, desc in self._available_devices] or ["0 — (not scanned)"]

    def _add_camera_entry(self, device_index: int = 0) -> None:
        """Add a camera row. device_index sets the initial device dropdown selection."""
        row_frame = ttk.Frame(self._cameras_container)
        row_frame.pack(fill="x", pady=2)

        enabled_var = tk.BooleanVar(value=True)
        label_var   = tk.StringVar(value=f"Camera {len(self._camera_entries)}")
        device_var  = tk.StringVar()

        # Pick the matching description or fall back to first entry
        devs = self._device_values()
        match_desc = next((d for _, d in self._available_devices
                           if _ == device_index), None)
        device_var.set(match_desc if match_desc else devs[0])

        entry: Dict = {
            "device_var":  device_var,
            "enabled_var": enabled_var,
            "label_var":   label_var,
            "row":         row_frame,
            "combo":       None,   # filled below
            "status_lbl":  None,   # filled below
        }
        # index property — derived from device_var at build-config time

        # ── Row layout ────────────────────────────────────────────────────────
        # [✓]  [device combo ▼]  [● status]  [label entry]  [✕]
        chk = ttk.Checkbutton(row_frame, variable=enabled_var,
                               command=lambda e=entry: self._on_toggle_camera_entry(e))
        chk.pack(side="left")

        combo = ttk.Combobox(row_frame, textvariable=device_var,
                              values=devs, width=12, state="readonly")
        combo.pack(side="left", padx=(2, 2))
        combo.bind("<<ComboboxSelected>>",
                   lambda _e, e=entry: self._on_device_changed(e))
        entry["combo"] = combo

        status_lbl = ttk.Label(row_frame, text="●", foreground="#7f849c", width=2)
        status_lbl.pack(side="left")
        entry["status_lbl"] = status_lbl

        ttk.Entry(row_frame, textvariable=label_var, width=9).pack(side="left", padx=2)

        ttk.Button(row_frame, text="✕", width=2,
                   command=lambda e=entry: self._on_remove_camera_entry(e)
                   ).pack(side="left", padx=(2, 0))

        self._camera_entries.append(entry)
        self._probe_entry_status(entry)

    def _parse_device_index(self, entry: Dict) -> int:
        """Extract the integer device index from an entry's device_var string."""
        raw = entry["device_var"].get()
        try:
            return int(raw.split(" — ")[0].strip())
        except (ValueError, IndexError):
            return 0

    def _probe_entry_status(self, entry: Dict) -> None:
        """Background-probe the selected device; update the status dot."""
        idx = self._parse_device_index(entry)
        def _probe():
            cap = cv2.VideoCapture(idx)
            ok  = cap.isOpened()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                # Update the combobox description with real resolution
                desc = f"{idx} — {w}\u00d7{h}"
            else:
                desc = f"{idx} — unavailable"
            # Update UI on main thread
            color  = "#a6e3a1" if ok else "#f38ba8"
            self.after(0, lambda: entry["status_lbl"].configure(
                foreground=color, text="●"))
            # Also refresh combobox value to show resolution
            self.after(0, lambda: entry["device_var"].set(desc))
        threading.Thread(target=_probe, daemon=True).start()

    def _on_device_changed(self, entry: Dict) -> None:
        """Re-probe status when user picks a different device."""
        entry["status_lbl"].configure(foreground="#f9e2af", text="●")  # probing…
        self._probe_entry_status(entry)

    def _on_add_camera(self) -> None:
        # Use the next unused index as default
        used = {self._parse_device_index(e) for e in self._camera_entries}
        nxt  = next((i for i in range(16) if i not in used), len(self._camera_entries))
        self._add_camera_entry(nxt)

    def _on_remove_camera_entry(self, entry: Dict) -> None:
        if len(self._camera_entries) <= 1:
            return
        entry["row"].destroy()
        self._camera_entries.remove(entry)

    def _on_scan_cameras(self) -> None:
        """Scan for available camera devices in a background thread."""
        self._scan_lbl.configure(text="scanning…", foreground="#f9e2af")
        def _do_scan():
            devices = scan_cameras(max_index=8)
            def _apply():
                if not devices:
                    self._scan_lbl.configure(text="no devices found",
                                              foreground="#f38ba8")
                    return
                self._available_devices = devices
                devs = self._device_values()
                self._scan_lbl.configure(
                    text=f"{len(devices)} device(s) found",
                    foreground="#a6e3a1")
                # Update all existing combos
                for e in self._camera_entries:
                    e["combo"].configure(values=devs)
                    # If current selection is the placeholder, swap to real first device
                    if "not scanned" in e["device_var"].get():
                        e["device_var"].set(devs[0])
                    self._probe_entry_status(e)
            self.after(0, _apply)
        threading.Thread(target=_do_scan, daemon=True).start()

    def _on_toggle_camera_entry(self, entry: Dict) -> None:
        if self._pipeline:
            idx = self._parse_device_index(entry)
            if entry["enabled_var"].get():
                self._pipeline.enable_camera(idx)
            else:
                self._pipeline.disable_camera(idx)

    # kept for any legacy callers
    def _on_toggle_camera(self, index: int, var: tk.BooleanVar) -> None:
        if self._pipeline:
            if var.get():
                self._pipeline.enable_camera(index)
            else:
                self._pipeline.disable_camera(index)

    # ── Video section ─────────────────────────────────────────────────────────

    def _build_video_section(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" Video File ", padding=6)
        lf.pack(fill="x", padx=4, pady=(0, 6))

        self._video_path_var = tk.StringVar(value="")
        ttk.Entry(lf, textvariable=self._video_path_var,
                  state="readonly").pack(fill="x")

        row = ttk.Frame(lf)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Browse…", command=self._on_browse_video).pack(side="left")
        ttk.Button(row, text="Clear",
                   command=lambda: self._video_path_var.set("")).pack(side="left", padx=4)

        self._loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Loop video", variable=self._loop_var).pack(anchor="w", pady=(4, 0))

    def _on_browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")]
        )
        if path:
            self._video_path_var.set(path)

    # ── Settings section ──────────────────────────────────────────────────────

    def _build_settings_section(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" Settings ", padding=6)
        lf.pack(fill="x", padx=4, pady=(0, 6))

        def row() -> ttk.Frame:
            r = ttk.Frame(lf); r.pack(fill="x", pady=2); return r

        r = row()
        ttk.Label(r, text="Device",     width=11).pack(side="left")
        self._device_var = tk.StringVar(value="cuda:0")
        ttk.Combobox(r, textvariable=self._device_var,
                     values=["cuda:0", "cuda:1", "cpu"],
                     width=9, state="readonly").pack(side="left")

        r = row()
        ttk.Label(r, text="Target FPS", width=11).pack(side="left")
        self._fps_var = tk.IntVar(value=30)
        ttk.Spinbox(r, from_=1, to=120, textvariable=self._fps_var, width=6).pack(side="left")

        r = row()
        ttk.Label(r, text="Skip frames",width=11).pack(side="left")
        self._skip_var = tk.IntVar(value=0)
        ttk.Spinbox(r, from_=0, to=10,  textvariable=self._skip_var, width=6).pack(side="left")
        ttk.Label(r, text="(0=every)", foreground="#7f849c",
                  font=("Segoe UI", 8)).pack(side="left", padx=4)

        r = row()
        ttk.Label(r, text="Confidence", width=11).pack(side="left")
        self._conf_var = tk.DoubleVar(value=0.3)
        ttk.Scale(r, variable=self._conf_var, from_=0.1, to=0.9,
                  orient="horizontal", length=90).pack(side="left")
        self._conf_lbl = ttk.Label(r, text="0.30", width=4)
        self._conf_lbl.pack(side="left")
        self._conf_var.trace_add("write",
            lambda *_: self._conf_lbl.configure(text=f"{self._conf_var.get():.2f}"))

    # ── WebSocket section ─────────────────────────────────────────────────────

    def _build_ws_section(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" WebSocket → Unity ", padding=6)
        lf.pack(fill="x", padx=4, pady=(0, 6))

        row = ttk.Frame(lf); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Port", width=6).pack(side="left")
        self._ws_port_var = tk.IntVar(value=8765)
        ttk.Spinbox(row, from_=1024, to=65535,
                    textvariable=self._ws_port_var, width=7).pack(side="left")

        self._lbl_ws_status  = ttk.Label(lf, text="● Stopped", foreground="#f38ba8")
        self._lbl_ws_status.pack(anchor="w", pady=(4, 0))
        self._lbl_ws_clients = ttk.Label(lf, text="Clients: 0", foreground="#7f849c")
        self._lbl_ws_clients.pack(anchor="w")

    # ── Manual trigger section ────────────────────────────────────────────────

    def _build_manual_trigger_section(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" Manual Trigger ", padding=6)
        lf.pack(fill="x", padx=4, pady=(0, 6))

        ttk.Label(lf, text="Template", foreground="#7f849c",
                  font=("Segoe UI", 8)).pack(anchor="w")

        combo_row = ttk.Frame(lf)
        combo_row.pack(fill="x", pady=(2, 4))

        self._trigger_name_var = tk.StringVar(value="")
        self._trigger_combo = ttk.Combobox(
            combo_row, textvariable=self._trigger_name_var,
            values=[], width=16,
        )
        self._trigger_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(combo_row, text="⟳", width=2,
                   command=self._refresh_template_names).pack(side="left", padx=(4, 0))

        conf_row = ttk.Frame(lf)
        conf_row.pack(fill="x", pady=(0, 6))
        ttk.Label(conf_row, text="Confidence", width=11).pack(side="left")
        self._trigger_conf_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(conf_row, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._trigger_conf_var,
                    format="%.2f", width=6).pack(side="left")

        self._btn_trigger = ttk.Button(
            lf, text="⚡  Send Signal",
            command=self._on_send_manual_match,
        )
        self._btn_trigger.pack(fill="x")
        self._lbl_trigger_status = ttk.Label(lf, text="", foreground="#7f849c",
                                             font=("Segoe UI", 8))
        self._lbl_trigger_status.pack(anchor="w", pady=(4, 0))

        # Populate template list on build
        self._refresh_template_names()

    def _refresh_template_names(self) -> None:
        """Scan the poseTemplate directory and update the trigger combobox."""
        template_dir = "poseTemplate"
        try:
            names = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(template_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
        except FileNotFoundError:
            names = []
        self._trigger_combo.configure(values=names)
        if names and not self._trigger_name_var.get():
            self._trigger_name_var.set(names[0])
        self._lbl_trigger_status.configure(
            text=f"{len(names)} template(s) found" if names else "No templates found",
            foreground="#a6e3a1" if names else "#f38ba8",
        )

    def _on_send_manual_match(self) -> None:
        name = self._trigger_name_var.get().strip()
        if not name:
            self._lbl_trigger_status.configure(
                text="Enter a template name first", foreground="#f9e2af")
            return
        if not self._pipeline or not self._pipeline.is_alive():
            self._lbl_trigger_status.configure(
                text="Pipeline not running", foreground="#f38ba8")
            return
        try:
            conf = float(self._trigger_conf_var.get())
        except (ValueError, tk.TclError):
            conf = 1.0
        conf = max(0.0, min(1.0, conf))
        self._pipeline.send_manual_match(name, conf)
        print(f"[ManualTrigger] Sent pose_match: template='{name}'  confidence={conf:.2f}")
        self._lbl_trigger_status.configure(
            text=f"Sent '{name}' @ {conf:.2f}", foreground="#a6e3a1")

    # ── Stats section ─────────────────────────────────────────────────────────

    def _build_stats_section(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text=" Live Stats ", padding=6)
        lf.pack(fill="x", padx=4, pady=(0, 6))
        self._lbl_fps    = ttk.Label(lf, text="FPS: —",
                                     font=("Segoe UI", 11, "bold"))
        self._lbl_fps.pack(anchor="w")
        self._lbl_people = ttk.Label(lf, text="People: —")
        self._lbl_people.pack(anchor="w")

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _build_config(self) -> Config:
        indices  = [self._parse_device_index(e) for e in self._camera_entries]
        disabled = [self._parse_device_index(e) for e in self._camera_entries
                    if not e["enabled_var"].get()]
        return Config(
            camera_indices   = indices,
            disabled_cameras = disabled,
            video_path       = self._video_path_var.get() or None,
            video_loop       = self._loop_var.get(),
            device           = self._device_var.get(),
            ws_port          = self._ws_port_var.get(),
            target_fps       = self._fps_var.get(),
            skip_frames      = self._skip_var.get(),
            kp_confidence    = round(self._conf_var.get(), 2),
        )

    def _on_start(self) -> None:
        if self._pipeline and self._pipeline.is_alive():
            return
        self._stop_event.clear()
        # drain queue
        while True:
            try: self._frame_queue.get_nowait()
            except queue.Empty: break

        cfg = self._build_config()
        self._pipeline = _PipelineThread(cfg, self._frame_queue, self._stop_event)
        self._pipeline.start()

        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._lbl_status.configure(text="● Loading model…", foreground="#f9e2af")
        self._lbl_ws_status.configure(text="● Starting…", foreground="#f9e2af")

    def _on_stop(self) -> None:
        self._stop_event.set()
        self._btn_stop.configure(state="disabled")
        self._lbl_status.configure(text="● Stopping…", foreground="#fab387")

    # ── UI tick (~30 fps polling) ─────────────────────────────────────────────

    def _ui_tick(self) -> None:
        # Check if pipeline died
        if self._pipeline and not self._pipeline.is_alive():
            if self._pipeline.error:
                self._lbl_status.configure(
                    text=f"● Error — see console", foreground="#f38ba8")
                print("[UI] Pipeline error:\n", self._pipeline.error)
            else:
                self._lbl_status.configure(text="● Idle", foreground="#a6e3a1")
            self._btn_start.configure(state="normal")
            self._btn_stop.configure(state="disabled")
            self._lbl_ws_status.configure(text="● Stopped", foreground="#f38ba8")
            self._pipeline = None

        # Drain frame queue — keep only the latest
        update: Optional[_FrameUpdate] = None
        try:
            while True:
                update = self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        if update is not None:
            # Safety check so the UI doesn't crash if grid is None
            if update.grid is not None:
                self._render_frame(update.grid)
                self._canvas.delete("placeholder")

            self._lbl_status.configure(text="● Running", foreground="#a6e3a1")
            self._lbl_fps.configure(text=f"FPS: {update.fps:.1f}")
            total = sum(update.people_counts.values())
            self._lbl_people.configure(text=f"People: {total}")
            self._lbl_ws_clients.configure(text=f"Clients: {update.ws_clients}")
            if update.ws_clients > 0:
                self._lbl_ws_status.configure(
                    text=f"● Live ({update.ws_clients} client{'s' if update.ws_clients != 1 else ''})",
                    foreground="#a6e3a1")
            elif self._pipeline:
                self._lbl_ws_status.configure(
                    text="● Waiting for Unity…", foreground="#f9e2af")

        self.after(33, self._ui_tick)

    def _render_frame(self, bgr: np.ndarray) -> None:
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = self._FEED_W, self._FEED_H

        h, w   = bgr.shape[:2]
        scale  = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=img)

        self._canvas.delete("frame")
        self._canvas.create_image(
            (cw - nw) // 2, (ch - nh) // 2,
            anchor="nw", image=self._photo, tags="frame"
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop_event.set()
        if self._pipeline:
            self._pipeline.join(timeout=3)
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = CameraTrackingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
