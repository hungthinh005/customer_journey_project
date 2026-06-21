"""
Email notifier.

Sends the agent-composed retention email over SMTP. In the local production
simulation this points at MailHog (catch-all inbox on http://localhost:8025),
so nothing ever reaches a real customer while testing.
"""

import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import settings


def send_email(to_address: str, subject: str, body: str) -> dict:
    """Send a plaintext email. Returns a status dict (never raises)."""
    if settings.agent_dry_run:
        return {"status": "dry_run", "error": None}

    msg = EmailMessage()
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_from_email}>"
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            if settings.smtp_use_tls:
                server.starttls()
            if settings.smtp_username:
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)
        return {"status": "sent", "error": None}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
