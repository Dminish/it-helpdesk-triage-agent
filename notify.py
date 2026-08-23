"""Email notification for raised tickets.

Uses stdlib smtplib, so there is no dependency to add. Configure with:

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=<app password, not your account password>
    SMTP_FROM="DanTech Helpdesk <you@gmail.com>"

With SMTP_HOST unset, nothing is sent and the message is returned unsent so the
caller can log or display it. That is the default: the app stays fully usable
without mail configured, rather than failing at the point a ticket is raised.
"""
import os
import smtplib
from email.message import EmailMessage

BODY = """Hi {name},

Your issue has been raised with the DanTech helpdesk and is now queued for a
human technician.

  Ticket      {ref}
  Category    {category}
  Raised for  {trigger}

What you reported:
  {issue}

A technician will follow up on this address. Quote {ref} if you need to chase
it.

DanTech IT Helpdesk
"""


def compose(to_email: str, name: str, ref: str, category: str, trigger: str, issue: str) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = f"[{ref}] Your DanTech helpdesk ticket"
    message["From"] = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER", "helpdesk@dantech.local")
    message["To"] = to_email
    message.set_content(
        BODY.format(name=name or "there", ref=ref, category=category, trigger=trigger, issue=issue)
    )
    return message


def send_ticket_email(
    to_email: str, name: str, ref: str, category: str, trigger: str, issue: str
) -> tuple[bool, str]:
    """Returns (sent, detail). Never raises: a mail failure must not lose the
    ticket, which is already recorded in the queue log by this point."""
    if not to_email:
        return False, "no email address on this ticket"

    message = compose(to_email, name, ref, category, trigger, issue)

    host = os.environ.get("SMTP_HOST")
    if not host:
        return False, "SMTP not configured, notification not sent"

    try:
        port = int(os.environ.get("SMTP_PORT", 587))
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            user = os.environ.get("SMTP_USER")
            # Google shows app passwords in four blocks of four; the spaces are
            # display formatting and SMTP auth fails if they are sent.
            password = "".join(os.environ.get("SMTP_PASSWORD", "").split())
            if user and password:
                server.login(user, password)
            server.send_message(message)
        return True, f"sent to {to_email}"
    except Exception as exc:  # network, auth, refused recipient
        return False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    # Self-check: composition works without any SMTP config, and sending is
    # correctly reported as skipped rather than crashing.
    msg = compose("user@example.com", "Dan", "DT-ABC123", "Network", "critical severity", "wifi down")
    assert msg["To"] == "user@example.com"
    assert "DT-ABC123" in msg["Subject"]
    assert "DT-ABC123" in msg.get_content()
    assert "Dan" in msg.get_content()

    os.environ.pop("SMTP_HOST", None)
    sent, detail = send_ticket_email("user@example.com", "Dan", "DT-ABC123", "Network", "critical", "wifi down")
    assert sent is False and "not configured" in detail, detail

    sent, detail = send_ticket_email("", "Dan", "DT-ABC123", "Network", "critical", "wifi down")
    assert sent is False and "no email" in detail, detail

    print("notify self-check passed")
