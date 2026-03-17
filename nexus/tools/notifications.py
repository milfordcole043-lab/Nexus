"""ntfy notification wrapper."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

NTFY_BASE_URL = "https://ntfy.sh"


async def send_notification(
    topic: str,
    title: str,
    message: str,
    priority: str = "default",
) -> bool:
    """Send a push notification via ntfy.sh.

    Returns True if sent successfully.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{NTFY_BASE_URL}/{topic}",
                content=message,
                headers={
                    "Title": title,
                    "Priority": priority,
                },
            )
            resp.raise_for_status()
            logger.info("Notification sent to topic '%s': %s", topic, title)
            return True
    except Exception as e:
        logger.error("Failed to send notification: %s", e)
        return False
