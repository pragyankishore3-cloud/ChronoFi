import logging
from typing import Dict, List, Union

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    Manages active WebSocket connections mapped to user IDs / client IDs.
    Supports multiple concurrent connections per client (e.g., across multiple tabs),
    and handles connection registration, deregistration, and personal alert delivery.
    """

    def __init__(self) -> None:
        # Maps client_id (str) -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_id: Union[str, int]) -> None:
        """
        Accepts the incoming WebSocket connection and registers it under the client_id.
        """
        await websocket.accept()
        key = str(client_id)
        if key not in self.active_connections:
            self.active_connections[key] = []
        self.active_connections[key].append(websocket)
        logger.info(
            "WebSocket connected for client_id: %s (Total active connections for client: %d)",
            key,
            len(self.active_connections[key]),
        )

    def disconnect(self, websocket: WebSocket, client_id: Union[str, int]) -> None:
        """
        Deregisters a WebSocket connection for the given client_id.
        Cleans up the map entry if no active connections remain for that client.
        """
        key = str(client_id)
        if key in self.active_connections:
            if websocket in self.active_connections[key]:
                self.active_connections[key].remove(websocket)
            if not self.active_connections[key]:
                del self.active_connections[key]
        logger.info("WebSocket disconnected for client_id: %s", key)

    async def send_personal_message(
        self, message: str, client_id: Union[str, int, WebSocket]
    ) -> None:
        """
        Sends a personal message string to the active WebSocket connection(s).
        Accepts client_id (str or int) or a direct WebSocket connection instance.
        Gracefully removes dead or abruptly closed sockets.
        """
        if isinstance(client_id, WebSocket):
            await client_id.send_text(message)
            return

        key = str(client_id)
        connections = self.active_connections.get(key, [])
        if not connections:
            logger.debug("No active WebSocket connection registered for client_id: %s", key)
            return

        stale_connections: List[WebSocket] = []
        for connection in list(connections):
            try:
                await connection.send_text(message)
            except Exception as exc:
                logger.warning(
                    "Failed to deliver message to client_id %s: %s. Marking connection for cleanup.",
                    key,
                    exc,
                )
                stale_connections.append(connection)

        for stale in stale_connections:
            self.disconnect(stale, key)

    def is_connected(self, client_id: Union[str, int]) -> bool:
        """
        Checks whether the given client_id currently has any active connections.
        """
        key = str(client_id)
        return key in self.active_connections and len(self.active_connections[key]) > 0


# Global singleton ConnectionManager instance
manager = ConnectionManager()


@router.websocket("/ws/alerts/{client_id}")
async def websocket_alerts_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time proactive autonomous financial alerts.
    Clients connect to /ws/alerts/{client_id} to receive streamed or pushed notifications.
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection open; receive client heartbeats, pings, or acknowledgments
            data = await websocket.receive_text()
            logger.debug("Received payload from client %s: %s", client_id, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as exc:
        logger.error("WebSocket error for client %s: %s", client_id, exc)
        manager.disconnect(websocket, client_id)


# Optional alias supporting /api/ws/alerts/{client_id} for frontend reverse-proxy consistency
@router.websocket("/api/ws/alerts/{client_id}")
async def websocket_alerts_api_alias(websocket: WebSocket, client_id: str):
    await websocket_alerts_endpoint(websocket, client_id)
