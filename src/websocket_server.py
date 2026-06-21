"""
websocket_server.py
--------------------
Async WebSocket server that streams pose data to Unity.
Completely resilient to client disconnects and reconnects.
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

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-server")
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        # ping_interval and ping_timeout ensure dead connections are aggressively cleared
        async with websockets.serve(
            self._handler,
            self._cfg.ws_host,
            self._cfg.ws_port,
            ping_interval=5,
            ping_timeout=5
        ):
            print(f"[WebSocket] Server running and waiting for Unity at ws://{self._cfg.ws_host}:{self._cfg.ws_port}")
            await asyncio.Future()  

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        addr = ws.remote_address
        print(f"[WebSocket] Unity Connected: {addr}")
        try:
            async for _ in ws:
                pass  
        except websockets.ConnectionClosed:
            pass # Silently handle abrupt closures
        finally:
            self._clients.discard(ws)
            print(f"[WebSocket] Unity Disconnected: {addr}. Ready for reconnection.")

    def send_pose(self, camera_id: int, bones: Dict[str, List[float]], keypoints: Optional[List] = None) -> None:
        if not self._loop or not self._clients:
            return # Skip silently if Unity is not connected
            
        payload: Dict[str, Any] = {"camera": camera_id, "bones": bones}
        if keypoints is not None:
            payload["keypoints"] = keypoints
            
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    def send_pose_match(self, template_name: str, confidence: float, camera_id: int = 0) -> None:
        if not self._loop or not self._clients:
            return
            
        payload: Dict[str, Any] = {
            "type": "pose_match",
            "template": template_name,
            "confidence": round(confidence, 4),
            "camera": camera_id,
        }
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload)
        dead: Set[WebSocketServerProtocol] = set()
        
        for ws in list(self._clients):
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
                
        # Clean up dead connections immediately
        self._clients -= dead
    
    @property
    def connected_clients(self) -> int:
        return len(self._clients)