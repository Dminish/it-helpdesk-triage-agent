"""Fake IT manual snippets, 4 per category, for demo retrieval."""

MANUALS = [
    # Hardware
    {"category": "Hardware", "text": (
        "Laptop won't power on: Hold the power button for 15 seconds to drain "
        "residual charge, then plug into AC power directly (skip the dock) and "
        "retry. If no LED activity, the battery or motherboard likely needs "
        "replacement — file a hardware ticket."
    )},
    {"category": "Hardware", "text": (
        "Printer shows 'paper jam' with no visible paper: Open every access "
        "panel including the rear duplexer, check for torn paper fragments near "
        "the rollers. Power cycle the printer after clearing. Recurring jams "
        "usually mean worn feed rollers, replace the roller kit."
    )},
    {"category": "Hardware", "text": (
        "External monitor not detected: Check the cable is seated at both ends, "
        "try a different port/cable to rule out a bad cable, and update the GPU "
        "driver. On docks, unplug/replug the dock's USB-C connector to force "
        "re-enumeration."
    )},
    {"category": "Hardware", "text": (
        "Keyboard keys sticking or unresponsive: Turn the device off, remove "
        "debris with compressed air, and if the issue persists after cleaning, "
        "swap the keyboard (built-in laptop keyboards are field-replaceable "
        "units)."
    )},
    # Network
    {"category": "Network", "text": (
        "Wi-Fi connects but no internet access: Forget the network and "
        "reconnect, flush DNS cache (ipconfig /flushdns), and check if a "
        "captive portal login is required. If other devices on the same "
        "network also fail, escalate to network team — possible AP or uplink "
        "issue."
    )},
    {"category": "Network", "text": (
        "VPN client fails to connect with 'handshake timeout': Confirm the "
        "user's local firewall isn't blocking UDP 500/4500, restart the VPN "
        "client service, and re-authenticate with fresh credentials. Repeated "
        "failures across many users indicate a VPN concentrator outage."
    )},
    {"category": "Network", "text": (
        "Slow network performance on wired connection: Check the switch port "
        "negotiated speed (should be 1Gbps, not 100Mbps), swap the ethernet "
        "cable, and rule out a faulty NIC by testing another port. Persistent "
        "slowness across a floor suggests switch congestion."
    )},
    {"category": "Network", "text": (
        "Cannot reach internal file share by hostname but IP works: This is a "
        "DNS resolution issue. Flush local DNS cache and confirm the client is "
        "pointed at the correct internal DNS server, not a public resolver."
    )},
    # Account
    {"category": "Account", "text": (
        "User locked out after failed login attempts: Verify identity via "
        "manager or badge, then unlock the account in the identity provider "
        "admin console. Advise the user to check for caps-lock or an outdated "
        "saved password in a browser autofill before retrying."
    )},
    {"category": "Account", "text": (
        "MFA push notifications not arriving: Confirm the phone has network "
        "connectivity, re-register the authenticator app if the device was "
        "recently reset, and generate a temporary bypass code if the user needs "
        "immediate access."
    )},
    {"category": "Account", "text": (
        "Password reset email never arrives: Check spam/junk folder first, "
        "confirm the account's recovery email is correct in the directory, and "
        "manually trigger a reset from the admin console if self-service fails "
        "repeatedly."
    )},
    {"category": "Account", "text": (
        "New hire has no access to required systems: Confirm the HR onboarding "
        "ticket has completed provisioning, check group membership in the "
        "identity provider, and manually add missing group assignments per the "
        "role's standard access template."
    )},
]
