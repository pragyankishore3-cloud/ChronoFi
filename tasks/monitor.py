import logging
from typing import Any, Dict, List, Optional, Union

from ai.background_graph import background_evaluator_graph
from api.websocket_routes import manager

logger = logging.getLogger(__name__)


async def mock_fetch_users_with_pending_limits() -> List[Dict[str, Any]]:
    """
    Simulates a database check that scans user portfolios to identify users with
    unutilized Section 80C tax deduction headroom approaching the fiscal deadline.

    In production, this queries SQLAlchemy tables:
        async with AsyncSessionLocal() as session:
            ...
    """
    return [
        {
            "user_id": 1,
            "name": "Arjun",
            "remaining_80c_limit": 50000.0,
            "financial_event": "Financial year-end approaching, user has ₹50k remaining in 80C",
        }
    ]


async def check_and_notify_tax_limits(
    user_id: Optional[Union[int, str]] = None,
    remaining_80c: Optional[float] = None,
    financial_event: Optional[str] = None,
) -> List[str]:
    """
    Background task that:
    1. Evaluates user tax state via a mock database check.
    2. Runs this structured payload through the headless LangGraph evaluator.
    3. Pushes the AI-generated advice string to active WebSocket connection(s) via ConnectionManager.
    """
    logger.info("Executing background task: check_and_notify_tax_limits...")

    # 1. Mock database check to retrieve candidates needing tax alerts
    if user_id is not None:
        pending_users = [
            {
                "user_id": user_id,
                "name": f"User {user_id}",
                "remaining_80c_limit": remaining_80c if remaining_80c is not None else 50000.0,
                "financial_event": financial_event or "Financial year-end approaching, user has ₹50k remaining in 80C",
            }
        ]
    else:
        pending_users = await mock_fetch_users_with_pending_limits()

    processed_alerts: List[str] = []

    for user_record in pending_users:
        uid = user_record["user_id"]
        client_id = str(uid)
        limit_val = user_record["remaining_80c_limit"]
        event_text = user_record["financial_event"]
        user_name = user_record.get("name", f"User {uid}")

        logger.info(
            "Invoking headless LangGraph for user_id=%s (remaining_80c=₹%s)...",
            uid,
            limit_val,
        )

        # 2. Invoke headless LangGraph evaluator with structured payload
        graph_payload = {
            "user_id": uid,
            "user_name": user_name,
            "financial_event": event_text,
            "remaining_80c_limit": limit_val,
        }

        graph_result = await background_evaluator_graph.ainvoke(graph_payload)
        ai_alert_message = graph_result.get("alert_message", "")

        if not ai_alert_message:
            logger.warning("No alert message produced by LangGraph for user_id=%s", uid)
            continue

        # 3. Push AI-generated string to active WebSocket connection
        logger.info(
            "Pushing generated alert to client_id=%s via WebSocket: %s",
            client_id,
            ai_alert_message,
        )
        await manager.send_personal_message(ai_alert_message, client_id=client_id)
        processed_alerts.append(ai_alert_message)

    # Additionally, if there are active connected WebSocket clients not in the mock list,
    # evaluate and deliver proactive alerts to them as well
    if user_id is None and manager.active_connections:
        mocked_client_ids = {str(u["user_id"]) for u in pending_users}
        for active_cid in list(manager.active_connections.keys()):
            if active_cid not in mocked_client_ids:
                logger.info(
                    "Detected active connection for client_id=%s. Generating alert...",
                    active_cid,
                )
                client_payload = {
                    "user_id": active_cid,
                    "user_name": f"User {active_cid}",
                    "financial_event": "Financial year-end approaching, user has ₹50k remaining in 80C",
                    "remaining_80c_limit": 50000.0,
                }
                res = await background_evaluator_graph.ainvoke(client_payload)
                msg = res.get("alert_message", "")
                if msg:
                    await manager.send_personal_message(msg, client_id=active_cid)
                    processed_alerts.append(msg)

    return processed_alerts
