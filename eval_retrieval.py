"""Calibrate CONFIDENCE_THRESHOLD against labelled retrieval cases.

Run: python eval_retrieval.py

Each case is a query the manuals either DO cover (should be answered) or do NOT
cover (should be escalated rather than guessed at). The script scores every
case, then sweeps candidate thresholds to find which one routes the most cases
correctly.

No LLM is involved: this exercises embeddings plus Pinecone only, so it runs in
seconds. Categories are given rather than predicted, to isolate retrieval from
classification (eval_classifier.py covers that half).
"""
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from graph import CONFIDENCE_THRESHOLD, retrieval_query, search_manuals

ANSWER = "answer"
ESCALATE = "escalate"

# (user turns, category, expected route)
# Multi-turn cases exercise the "bare follow-up" path, where the last message
# alone carries no topic words.
CASES = [
    # --- Manuals cover these: retrieval should be confident enough to answer.
    (["my laptop wont power on at all"], "Hardware", ANSWER),
    (["printer keeps saying paper jam but theres no paper"], "Hardware", ANSWER),
    (["my second screen isnt detected through the dock"], "Hardware", ANSWER),
    (["some keys on my keyboard stick and dont register"], "Hardware", ANSWER),
    (["wifi says connected but nothing loads"], "Network", ANSWER),
    (["vpn wont connect, handshake keeps timing out"], "Network", ANSWER),
    (["my ethernet connection is crawling today"], "Network", ANSWER),
    (["i can reach the file share by ip but not by name"], "Network", ANSWER),
    (["im locked out after typing my password wrong too many times"], "Account", ANSWER),
    (["mfa prompts never reach my phone"], "Account", ANSWER),
    (["the password reset email hasnt arrived"], "Account", ANSWER),
    (["our new starter cant get into any of the systems"], "Account", ANSWER),
    # Bare follow-ups: only answerable because earlier turns supply the topic.
    (["vpn keeps timing out", "i tried that, still failing"], "Network", ANSWER),
    (["my keyboard keys are sticking", "did that, no better"], "Hardware", ANSWER),
    (["locked out of my account", "that didnt work either"], "Account", ANSWER),
    # --- Manuals do not cover these: escalating beats inventing an answer.
    (["my desk chair squeaks when i lean back"], "Hardware", ESCALATE),
    (["the coffee machine on our floor is broken"], "Hardware", ESCALATE),
    (["i need a second monitor ordered for my new desk"], "Hardware", ESCALATE),
    (["how do i book a meeting room for next tuesday"], "Network", ESCALATE),
    (["can we get a licence for adobe illustrator"], "Account", ESCALATE),
    (["what is the wifi password for the guest network"], "Network", ESCALATE),
    (["i want to expense a usb hub i bought myself"], "Account", ESCALATE),
    (["please delete my old teams recordings from last year"], "Account", ESCALATE),
]


def route(score: float, threshold: float) -> str:
    return ESCALATE if score < threshold else ANSWER


def main():
    scored = []
    for turns, category, expected in CASES:
        messages = [HumanMessage(t) for t in turns]
        _, score = search_manuals(retrieval_query(messages), category)
        scored.append((turns[-1], category, expected, score))

    print("per-case scores")
    print("-" * 72)
    for last_turn, category, expected, score in sorted(scored, key=lambda r: -r[3]):
        mark = "ok " if route(score, CONFIDENCE_THRESHOLD) == expected else "MISS"
        print(f"  {score:.3f}  {mark}  want {expected:8} {category:8} {last_turn[:38]}")

    answerable = [s for _, _, e, s in scored if e == ANSWER]
    unanswerable = [s for _, _, e, s in scored if e == ESCALATE]
    print()
    print(f"answerable   min {min(answerable):.3f}  max {max(answerable):.3f}")
    print(f"unanswerable min {min(unanswerable):.3f}  max {max(unanswerable):.3f}")
    gap = min(answerable) - max(unanswerable)
    if gap > 0:
        print(f"clean separation, gap {gap:.3f}  (any threshold inside it routes 100%)")
    else:
        print(f"OVERLAP of {-gap:.3f}: no threshold separates these perfectly")

    print()
    print("threshold sweep")
    print("-" * 72)
    results = []
    for step in range(20, 81):
        threshold = step / 100
        hits = sum(route(s, threshold) == e for _, _, e, s in scored)
        results.append((hits, threshold))
    best = max(hits for hits, _ in results)
    best_range = [t for hits, t in results if hits == best]

    shown = {round(t, 2) for t in (min(best_range), max(best_range), CONFIDENCE_THRESHOLD)}
    for hits, threshold in results:
        if round(threshold, 2) in shown:
            tag = " <- current" if threshold == CONFIDENCE_THRESHOLD else ""
            print(f"  {threshold:.2f}  {hits}/{len(scored)} correct{tag}")

    current = next(h for h, t in results if t == CONFIDENCE_THRESHOLD)
    print()
    print(f"current {CONFIDENCE_THRESHOLD}: {current}/{len(scored)} correct")
    print(f"best    {best}/{len(scored)} correct at thresholds "
          f"{min(best_range):.2f}-{max(best_range):.2f}")
    if best > current:
        midpoint = (min(best_range) + max(best_range)) / 2
        print(f"suggest CONFIDENCE_THRESHOLD = {midpoint:.2f} (middle of the best range)")


if __name__ == "__main__":
    main()
