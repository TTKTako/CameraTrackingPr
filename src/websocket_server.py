"""
websocket_server.py
--------------------
Async WebSocket server that streams pose data to Unity (or any WS client).

WebSocket payload (JSON, sent once per frame per camera):
{
  "camera"   : <int>          camera index
  "bones"    : {              VRM HumanBodyBones → [x, y, z, w] quaternion
    "Hips"       : [x,y,z,w],
    "Spine"      : [x,y,z,w],
    ...
  },
  "keypoints": [[x,y], ...]   raw 2-D keypoints (K×2), optional
}

Thread safety
─────────────
The main OpenCV loop runs on the main thread; this server runs its asyncio
event loop on a dedicated daemon thread.  `send_pose()` is the only public
method called from the main thread — it is thread-safe.
"""

import asyncio
import json
import threading
from typing import Any, Dict, List, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from .config import Config


class PoseWebSocketServer:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._clients: Set[WebSocketServerProtocol] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._no_client_warned: bool = False   # log once when data is dropped

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the WebSocket server on a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with websockets.serve(
            self._handler,
            self._cfg.ws_host,
            self._cfg.ws_port,
        ):
            print(
                f"[WebSocket] Listening on "
                f"ws://{self._cfg.ws_host}:{self._cfg.ws_port}"
            )
            await asyncio.Future()  # run indefinitely

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        self._no_client_warned = False   # reset so the warning fires again if they disconnect
        addr = ws.remote_address
        print(f"[WebSocket] Client connected    : {addr}")
        try:
            async for _ in ws:
                pass  # one-way stream; ignore any incoming messages
        finally:
            self._clients.discard(ws)
            print(f"[WebSocket] Client disconnected : {addr}")

    # ── Public API (thread-safe) ──────────────────────────────────────────────

    def send_pose(
        self,
        camera_id: int,
        bones: Dict[str, List[float]],
        keypoints: Optional[List] = None,
    ) -> None:
        """
        Enqueue a pose message for broadcasting to all connected clients.
        Safe to call from the main (OpenCV) thread.
        """
        if not self._loop or not self._clients:
            if not self._no_client_warned:
                print(
                    f"[WebSocket] WARNING — pose data ready but no client connected. "
                    f"Connect Unity to ws://{self._cfg.ws_host}:{self._cfg.ws_port}"
                )
                self._no_client_warned = True
            return
        payload: Dict[str, Any] = {"camera": camera_id, "bones": bones}
        if keypoints is not None:
            payload["keypoints"] = keypoints
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    def send_pose_match(
        self,
        template_name: str,
        confidence: float,
        camera_id: int = 0,
    ) -> None:
        """
        Notify Unity that the user is holding a recognised pose.

        Payload::

            {
              "type"      : "pose_match",
              "template"  : "<stem of the template image filename>",
              "confidence": 0.87,
              "camera"    : 0
            }

        Safe to call from the main (OpenCV) thread.
        """
        if not self._loop or not self._clients:
            return
        payload: Dict[str, Any] = {
            "type":       "pose_match",
            "template":   template_name,
            "confidence": round(confidence, 4),
            "camera":     camera_id,
        }
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._clients:
            return
        message = json.dumps(payload)
        dead: Set[WebSocketServerProtocol] = set()
        for ws in list(self._clients):
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead

    @property
    def connected_clients(self) -> int:
        return len(self._clients)
