"""
test-sent-data.py
-----------------
WebSocket debug client — connects to the running CameraTrackingPose app and
prints every message it receives in a readable, colour-coded format.

Usage
─────
    python test-sent-data.py                  # default ws://localhost:8765
    python test-sent-data.py --port 8766
    python test-sent-data.py --host 192.168.1.10 --port 8765
    python test-sent-data.py --raw            # print raw JSON, no formatting
    python test-sent-data.py --no-pose        # hide regular pose frames (bones/kps)
    python test-sent-data.py --no-match       # hide pose_match events

Known message types
───────────────────
  Regular pose frame  {"camera": 0, "bones": {...}, "keypoints": [...]}
  Pose match event    {"type": "pose_match", "template": "...", "confidence": 0.87, "camera": 0}
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict

try:
    import websockets
except ImportError:
    print("websockets not installed — run:  pip install websockets")
    sys.exit(1)

# ── ANSI colours (disabled on Windows if colorama not present) ────────────────
try:
    import colorama
    colorama.init()
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _CYAN   = "\033[36m"
    _RED    = "\033[31m"
    _DIM    = "\033[2m"
    _MAGENTA= "\033[35m"
except ImportError:
    _RESET = _BOLD = _GREEN = _YELLOW = _CYAN = _RED = _DIM = _MAGENTA = ""


# ── Counters ──────────────────────────────────────────────────────────────────
_stats: Dict[str, int] = {"pose": 0, "match": 0, "unknown": 0, "bytes": 0}
_start_time: float = 0.0


def _fmt_quat(q: list) -> str:
    """Format a [x, y, z, w] quaternion compactly."""
    return f"[{q[0]:+.3f}  {q[1]:+.3f}  {q[2]:+.3f}  {q[3]:+.3f}]"


def _print_pose_frame(msg: Dict[str, Any], show_bones: bool, show_kps: bool) -> None:
    cam   = msg.get("camera", "?")
    bones = msg.get("bones", {})
    kps   = msg.get("keypoints", [])

    print(f"{_CYAN}{_BOLD}[POSE FRAME]{_RESET}  cam={cam}  "
          f"bones={len(bones)}  keypoints={len(kps)}")

    if show_bones and bones:
        for name, q in sorted(bones.items()):
            print(f"  {_DIM}{name:<20}{_RESET} {_fmt_quat(q)}")

    if show_kps and kps:
        print(f"  {_DIM}keypoints (first 5):{_RESET}")
        for i, pt in enumerate(kps[:5]):
            print(f"    [{i:2d}]  x={pt[0]:7.1f}  y={pt[1]:7.1f}")
        if len(kps) > 5:
            print(f"    {_DIM}... {len(kps) - 5} more{_RESET}")


def _print_match_event(msg: Dict[str, Any]) -> None:
    template   = msg.get("template", "?")
    confidence = msg.get("confidence", 0.0)
    cam        = msg.get("camera", "?")
    bar_filled = int(confidence * 20)
    bar        = "█" * bar_filled + "░" * (20 - bar_filled)
    print(
        f"\n{_GREEN}{_BOLD}★ POSE MATCH ★{_RESET}  "
        f"template={_YELLOW}{template}{_RESET}  "
        f"cam={cam}\n"
        f"  confidence  [{_GREEN}{bar}{_RESET}]  {confidence:.1%}\n"
    )


def _print_stats() -> None:
    elapsed = time.perf_counter() - _start_time
    fps = _stats["pose"] / elapsed if elapsed > 0 else 0.0
    kb  = _stats["bytes"] / 1024
    print(
        f"\n{_DIM}─── stats ───  "
        f"pose={_stats['pose']}  match={_stats['match']}  "
        f"unknown={_stats['unknown']}  "
        f"{kb:.1f} KB  {fps:.1f} fps  {elapsed:.0f}s{_RESET}"
    )


async def listen(
    host: str,
    port: int,
    raw: bool,
    show_pose: bool,
    show_match: bool,
    show_bones: bool,
    show_kps: bool,
    stats_interval: float,
) -> None:
    global _start_time
    uri = f"ws://{host}:{port}"
    print(f"{_BOLD}Connecting to {uri} …{_RESET}  (Ctrl-C to quit)\n")

    last_stats = time.perf_counter()
    _start_time = last_stats

    async for ws in websockets.connect(uri, ping_interval=None):
        try:
            print(f"{_GREEN}Connected.{_RESET}\n")
            async for raw_msg in ws:
                _stats["bytes"] += len(raw_msg)

                if raw:
                    print(raw_msg)
                    continue

                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    print(f"{_RED}[INVALID JSON]{_RESET} {raw_msg[:200]}")
                    _stats["unknown"] += 1
                    continue

                msg_type = msg.get("type", "pose")

                if msg_type == "pose_match":
                    _stats["match"] += 1
                    if show_match:
                        _print_match_event(msg)

                elif msg_type == "pose":
                    _stats["pose"] += 1
                    if show_pose:
                        _print_pose_frame(msg, show_bones=show_bones, show_kps=show_kps)

                else:
                    _stats["unknown"] += 1
                    print(f"{_MAGENTA}[UNKNOWN type={msg_type}]{_RESET} {str(msg)[:200]}")

                # Periodic stats line
                now = time.perf_counter()
                if stats_interval > 0 and now - last_stats >= stats_interval:
                    _print_stats()
                    last_stats = now

        except websockets.ConnectionClosed as exc:
            print(f"\n{_YELLOW}Connection closed ({exc.code}).  Reconnecting…{_RESET}\n")
            await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="CameraTrackingPose WebSocket debug client")
    parser.add_argument("--host",           default="localhost",
                        help="Server host (default: localhost)")
    parser.add_argument("--port",           type=int, default=8765,
                        help="Server port (default: 8765)")
    parser.add_argument("--raw",            action="store_true",
                        help="Print raw JSON without formatting")
    parser.add_argument("--no-pose",        action="store_true",
                        help="Hide regular pose frame messages")
    parser.add_argument("--no-match",       action="store_true",
                        help="Hide pose_match events")
    parser.add_argument("--bones",          action="store_true",
                        help="Show individual bone quaternions in pose frames")
    parser.add_argument("--kps",            action="store_true",
                        help="Show keypoint coordinates in pose frames")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        metavar="SECS",
                        help="Print stats every N seconds; 0 to disable (default: 5)")
    args = parser.parse_args()

    try:
        asyncio.run(listen(
            host           = args.host,
            port           = args.port,
            raw            = args.raw,
            show_pose      = not args.no_pose,
            show_match     = not args.no_match,
            show_bones     = args.bones,
            show_kps       = args.kps,
            stats_interval = args.stats_interval,
        ))
    except KeyboardInterrupt:
        _print_stats()
        print(f"\n{_DIM}Bye.{_RESET}")


if __name__ == "__main__":
    main()
