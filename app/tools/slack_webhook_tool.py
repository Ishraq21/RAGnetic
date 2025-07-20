import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SlackWebhookTool:
    """
    A tool to send messages to a Slack channel via a webhook.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def run(self, webhook_url: str, message: str, username: str = "RAGnetic", icon_emoji: str = ":robot_face:") -> Dict[
        str, Any]:
        """
        Sends a message to a Slack channel using a webhook URL.

        Args:
            webhook_url (str): The incoming webhook URL for the Slack channel.
            message (str): The text message to send.
            username (str): The username to post the message as.
            icon_emoji (str): The emoji to use as the bot's icon.

        Returns:
            Dict: A dictionary containing the response status and message.
        """
        if not webhook_url or not message:
            return {"error": "Webhook URL and message are required."}

        payload = {
            "text": message,
            "username": username,
            "icon_emoji": icon_emoji
        }

        try:
            logger.info(f"Sending message to Slack via webhook.")
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            if response.text == "ok":
                return {"status": "success", "message": "Message sent to Slack successfully."}
            else:
                return {"status": "failure", "message": response.text}

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack message: {e}")
            return {"status": "failure", "message": str(e)}