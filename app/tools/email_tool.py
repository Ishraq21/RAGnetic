import os
import smtplib
import ssl
import logging
import certifi
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, Type

from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class EmailToolInput(BaseModel):
    """Input schema for the EmailTool."""
    to_email: str = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The HTML or plain text content of the email.")


class EmailTool:
    """A tool for sending emails via a secure SMTP server."""
    name: str = "email_tool"
    description: str = (
        "Sends an email to a specified recipient with a subject and body. "
        "Uses the certifi CA bundle for SSL verification."
    )
    args_schema: Type[BaseModel] = EmailToolInput

    def __init__(self):
        # Load configuration securely from environment variables
        self.smtp_host = os.environ.get("SMTP_HOST")
        self.smtp_port = int(os.environ.get("SMTP_PORT", 465))
        self.smtp_username = os.environ.get("SMTP_USERNAME")
        self.smtp_password = os.environ.get("SMTP_PASSWORD")

        missing = [k for k in ("SMTP_HOST","SMTP_PORT","SMTP_USERNAME","SMTP_PASSWORD")
                   if not os.environ.get(k)]
        if missing:
            raise ValueError(
                f"SMTP configuration is incomplete. Missing: {', '.join(missing)}"
            )

    def run(self, to_email: str, subject: str, body: str, **kwargs: Any) -> Dict[str, str]:
        logger.info(f"Preparing to send email to: {to_email} with subject: '{subject}'")

        # Build the MIME message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.smtp_username
        message["To"] = to_email
        message.attach(MIMEText(body, "plain"))
        message.attach(MIMEText(body, "html"))

        # Create an SSL context that uses certifi's CA bundle
        try:
            context = ssl.create_default_context(cafile=certifi.where())
        except Exception as e:
            logger.warning(f"Could not load certifi bundle, falling back to system default: {e}")
            context = ssl.create_default_context()

        try:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.smtp_username, to_email, message.as_string())

            success_message = f"Email sent successfully to {to_email}."
            logger.info(success_message)
            return {"status": "success", "message": success_message}

        except smtplib.SMTPException as e:
            error_message = f"SMTP Error: {e}"
            logger.error(f"Failed to send email via SMTP: {e}", exc_info=True)
            return {"status": "error", "message": error_message}

        except ssl.SSLCertVerificationError as e:
            error_message = (
                "SSL certificate verification failed. "
                "Make sure you have certifi installed (`pip install certifi`) "
                "or that your system’s CA bundle is up to date."
            )
            logger.error(error_message + f" Details: {e}", exc_info=True)
            return {"status": "error", "message": error_message}

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            logger.error(f"An unexpected error occurred while sending email: {e}", exc_info=True)
            return {"status": "error", "message": error_message}
