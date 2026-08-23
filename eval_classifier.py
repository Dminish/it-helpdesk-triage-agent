"""Accuracy check for the triage classifier.

Run: python eval_classifier.py

Prints per-field accuracy and lists every miss. The classifier is a small local
model, so expect some variance between runs; use this to compare prompt changes,
not as a pass/fail gate.
"""
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from graph import classifier

# (messages, expected_category, expected_severity)
# Single-turn cases pass one user message. Follow-up cases pass a short history
# to check that context does not flip the classification.
CASES = [
    # Hardware, normal
    ([HumanMessage("my keyboard keys are sticking and some dont respond")], "Hardware", "normal"),
    ([HumanMessage("laptop wont power on at all, no lights")], "Hardware", "normal"),
    ([HumanMessage("printer says paper jam but i cant see any paper")], "Hardware", "normal"),
    ([HumanMessage("second monitor isnt being detected when i dock")], "Hardware", "normal"),
    # Network, normal
    ([HumanMessage("vpn keeps timing out cant connect")], "Network", "normal"),
    ([HumanMessage("wifi connects but no internet on my machine")], "Network", "normal"),
    ([HumanMessage("my wired connection is really slow today")], "Network", "normal"),
    # Account, normal
    ([HumanMessage("locked out after too many failed logins")], "Account", "normal"),
    ([HumanMessage("mfa push notifications never arrive on my phone")], "Account", "normal"),
    ([HumanMessage("password reset email never showed up")], "Account", "normal"),
    ([HumanMessage("new starter has no access to the finance system")], "Account", "normal"),
    # Critical
    ([HumanMessage("the production server room caught fire")], "Hardware", "critical"),
    ([HumanMessage("prod database is completely down for everyone")], "Network", "critical"),
    ([HumanMessage("entire marketing department cannot access the network")], "Network", "critical"),
    ([HumanMessage("we think an attacker has admin access to the domain controller")], "Account", "critical"),
    # Scope is implied by naming a team/site, not stated with a word like "entire".
    ([HumanMessage("network down in the marketing department")], "Network", "critical"),
    ([HumanMessage("nobody on the 3rd floor can print")], "Hardware", "critical"),
    # Follow-ups: must inherit the original issue, not escalate on repetition
    (
        [
            HumanMessage("vpn keeps timing out cant connect"),
            AIMessage("Check that UDP 500 and 4500 are not blocked, then restart the VPN client."),
            HumanMessage("i tried that, still failing"),
        ],
        "Network",
        "normal",
    ),
    (
        [
            HumanMessage("my keyboard keys are sticking"),
            AIMessage("Power off, clear debris with compressed air, then retest."),
            HumanMessage("did that and its no better"),
        ],
        "Hardware",
        "normal",
    ),
]


def main():
    cat_hits = sev_hits = both_hits = 0
    misses = []

    for messages, want_cat, want_sev in CASES:
        result = classifier.invoke({"messages": messages})
        cat_ok = result.category == want_cat
        sev_ok = result.severity == want_sev
        cat_hits += cat_ok
        sev_hits += sev_ok
        both_hits += cat_ok and sev_ok
        if not (cat_ok and sev_ok):
            misses.append((messages[-1].content, want_cat, want_sev, result.category, result.severity))

    total = len(CASES)
    print(f"category  {cat_hits}/{total}  ({cat_hits / total:.0%})")
    print(f"severity  {sev_hits}/{total}  ({sev_hits / total:.0%})")
    print(f"both      {both_hits}/{total}  ({both_hits / total:.0%})")

    if misses:
        print(f"\n{len(misses)} miss(es):")
        for issue, want_cat, want_sev, got_cat, got_sev in misses:
            print(f"  {issue!r}")
            print(f"    want {want_cat}/{want_sev}  got {got_cat}/{got_sev}")


if __name__ == "__main__":
    main()
