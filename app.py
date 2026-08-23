import base64
import os
import pathlib
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Community Cloud supplies config through st.secrets rather than a
# .env file. Mirror it into the environment before graph.py reads it at import.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # no secrets.toml locally, which is the normal case
    pass

from langchain_core.messages import HumanMessage

from graph import graph, load_history, ticket_ref

st.set_page_config(
    page_title="DanTech IT Helpdesk",
    page_icon=":material/support_agent:",
    layout="centered",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* Palette taken from the logo: navy wordmark, teal secondary, white ground.
   Committed to a single light theme rather than shipping a half-tuned dark
   variant, since the logo itself is drawn for a light background. */
:root {
  --bg: #F5F8FA;
  --surface: #FFFFFF;
  --surface-2: #EDF3F6;
  --border: rgba(18,52,77,0.14);
  --border-strong: rgba(18,52,77,0.26);
  --text: #12344D;
  --muted: #5D7C8E;
  --accent: #1798AD;
  --accent-deep: #1B4F72;
  --accent-ink: #FFFFFF;
  --critical: #C4453B;
  --critical-dim: rgba(196,69,59,0.10);
  --accent-dim: rgba(23,152,173,0.10);
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--bg); }
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
.block-container { max-width: 760px; padding-top: 2.2rem; }

/* ---- brand bar ---- */
.brandbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding-bottom: 0.9rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.4rem;
}
.brandbar .mark { flex: 0 0 auto; height: 38px; width: auto; }
.brandhero .logo { height: 132px; width: auto; margin-bottom: 0.5rem; }
.brandhero .names { display: none; }
.brandhero .mark { height: 64px; }
.brandbar .names { display: flex; flex-direction: column; line-height: 1.25; }
.brandbar .wordmark {
  font-weight: 600;
  font-size: 1.02rem;
  color: var(--text);
  letter-spacing: -0.01em;
}
.brandbar .product {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
}
.brandbar .ticket {
  margin-left: auto;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  color: var(--accent);
  border: 1px solid rgba(23,152,173,0.40);
  border-radius: 6px;
  padding: 0.25rem 0.55rem;
}

.lede {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.55;
  max-width: 60ch;
  margin-bottom: 1.6rem;
}

/* ---- chat ---- */
[data-testid="stChatMessage"] { background: transparent; border: none; padding: 0.4rem 0; }
[data-testid="stChatMessageContent"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1.1rem;
}
[data-testid="stChatMessageAvatarUser"] { background: var(--surface-2) !important; color: var(--text) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent-dim) !important; color: var(--accent) !important; }

.meta-row {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.74rem;
  letter-spacing: 0.04em;
  color: var(--muted);
  margin-top: 0.6rem;
  text-transform: uppercase;
}
.meta-row span { padding-left: 0.9rem; border-left: 1px solid var(--border); }
.meta-row span:first-child { padding-left: 0; border-left: none; }
.meta-row .tag-category { color: var(--text); }
.meta-row .tag-critical { color: var(--critical); }

.panel {
  padding-left: 0.9rem;
  border-left: 3px solid var(--accent);
  color: var(--text);
  font-size: 0.95rem;
  line-height: 1.55;
}
.panel.escalated { border-left-color: var(--critical); }
.panel-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.35rem;
  display: block;
}
.panel.escalated .panel-label { color: var(--critical); }
.panel code {
  background: var(--surface-2);
  color: var(--text);
  padding: 0.1em 0.35em;
  border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.88em;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .sb-heading {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.9rem 0 0.7rem 0;
  padding-top: 1.1rem;
  border-top: 1px solid var(--border);
}
[data-testid="stSidebar"] .sb-heading.first {
  margin-top: 0.4rem;
  padding-top: 0;
  border-top: none;
}
[data-testid="stSidebar"] .sb-item {
  color: var(--text);
  font-size: 0.88rem;
  line-height: 1.5;
  padding: 0.32rem 0;
}
[data-testid="stSidebar"] .sb-item small { color: var(--muted); display: block; font-size: 0.78rem; }

/* ---- input ---- */
[data-testid="stChatInput"] {
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  background: var(--surface-2);
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 6px !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--accent-ink) !important; }

[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }

/* ---- empty state ---- */
.empty-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0.6rem 0 0.7rem 0;
}
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button {
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 400;
  padding: 0.6rem 0.9rem;
  justify-content: flex-start !important;
}
/* Streamlit nests a full-width flex div inside the button and centres there, so
   styling the button element alone has no effect on the label. */
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button > div {
  justify-content: flex-start !important;
  text-align: left !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button p {
  text-align: left !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover {
  border-color: rgba(23,152,173,0.55);
  color: var(--accent);
}

/* ---- motion (intensity 7: entry reveals and tactile feedback, no hijacking) ---- */
@keyframes riseIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: none; }
}
@keyframes drawRule {
  from { transform: scaleY(0); }
  to   { transform: scaleY(1); }
}
[data-testid="stChatMessage"] {
  animation: riseIn 0.44s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.panel {
  position: relative;
  animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.06s both;
}
.panel::before {
  content: "";
  position: absolute;
  left: -3px; top: 0; bottom: 0;
  width: 3px;
  background: inherit;
  transform-origin: top;
  animation: drawRule 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.meta-row { animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.16s both; }
.brandbar { animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }
.lede { animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.08s both; }

/* Staggered so the example prompts cascade rather than snapping in together. */
[data-testid="stMainBlockContainer"] [data-testid="stButton"] {
  animation: riseIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stMainBlockContainer"] [data-testid="stButton"]:nth-of-type(1) { animation-delay: 0.06s; }
[data-testid="stMainBlockContainer"] [data-testid="stButton"]:nth-of-type(2) { animation-delay: 0.12s; }
[data-testid="stMainBlockContainer"] [data-testid="stButton"]:nth-of-type(3) { animation-delay: 0.18s; }

[data-testid="stMainBlockContainer"] [data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
.brandbar .ticket {
  transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1),
              border-color 0.22s ease, background 0.22s ease, color 0.22s ease;
}
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); }
[data-testid="stMainBlockContainer"] [data-testid="stButton"] button:active,
[data-testid="stFormSubmitButton"] button:active { transform: translateY(0) scale(0.995); }

@media (prefers-reduced-motion: reduce) {
  [data-testid="stChatMessage"], .panel, .panel::before, .meta-row, .brandbar,
  .lede, [data-testid="stMainBlockContainer"] [data-testid="stButton"] {
    animation: none !important;
  }
  [data-testid="stMainBlockContainer"] [data-testid="stButton"] button:hover,
  [data-testid="stFormSubmitButton"] button:hover { transform: none; }
}

/* ---- landing ---- */
.welcome { padding-top: 6vh; }
.welcome .hero-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.9rem;
}
.welcome h1 { animation: riseIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.05s both; }
.welcome p { animation: riseIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.12s both; }
.formcard-title {
  font-weight: 600;
  margin-bottom: 0.9rem;
  color: var(--text);
  font-size: 1rem;
  margin-bottom: 0.2rem;
}
@media (prefers-reduced-motion: reduce) {
  .welcome h1, .welcome p, [data-testid="stForm"] { animation: none !important; }
}
@media (max-width: 768px) {
  .welcome { padding-top: 2.5vh; }
  .brandhero .logo { height: 56px; }
}
.welcome h1 {
  font-size: 2rem;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.015em;
  margin: 1.2rem 0 0.6rem 0;
}
.welcome p {
  color: var(--muted);
  font-size: 0.97rem;
  line-height: 1.6;
  max-width: 52ch;
  margin-bottom: 0.4rem;
}
.welcome .steps {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.74rem;
  letter-spacing: 0.05em;
  color: var(--muted);
  text-transform: uppercase;
  margin: 1.6rem 0 0.4rem 0;
  padding-top: 1.1rem;
  border-top: 1px solid var(--border);
}
[data-testid="stTextInput"] input {
  background: var(--surface-2);
  border: 1px solid var(--border-strong);
  color: var(--text);
  border-radius: 8px;
}
[data-testid="stTextInput"] label { color: var(--muted) !important; font-size: 0.85rem; }
[data-testid="stForm"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.4rem 0.9rem 1.4rem;
  animation: riseIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.18s both;
}
[data-testid="stFormSubmitButton"] button {
  background: var(--accent);
  color: var(--accent-ink);
  border: none;
  border-radius: 8px;
  font-weight: 600;
}
[data-testid="stFormSubmitButton"] button:hover { background: #147F91; }
.privacy {
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.5;
  margin-top: 1rem;
  max-width: 52ch;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# The mark encodes what the product does rather than just its initial: one input
# arriving from the left, a decision point, and two outcomes -- resolved (a
# closed dot) and escalated (an open arm in the critical colour). A generic "D"
# would say nothing about routing, which is the whole argument of the product.
# Kept on one line: st.markdown renders indented HTML as a code block.
MARK = (
    '<svg class="mark" width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">'
    '<rect x="0.75" y="0.75" width="32.5" height="32.5" rx="8" fill="rgba(23,152,173,0.10)" '
    'stroke="rgba(23,152,173,0.40)" stroke-width="1.5"/>'
    '<path d="M8 17h6.5" stroke="#1B4F72" stroke-width="2.2" stroke-linecap="round"/>'
    '<path d="M14.5 17c3.4 0 3.6-5 7-5" stroke="#1798AD" stroke-width="2.2" '
    'stroke-linecap="round" fill="none"/>'
    '<path d="M14.5 17c3.4 0 3.6 5 7 5" stroke="#C4453B" stroke-width="2.2" '
    'stroke-linecap="round" fill="none"/>'
    '<circle cx="24.6" cy="12" r="2.4" fill="#1798AD"/>'
    '<circle cx="24.6" cy="22" r="2.4" fill="none" stroke="#C4453B" stroke-width="2.2"/>'
    "</svg>"
)

ASSETS = pathlib.Path(__file__).parent / "assets"
# The full lockup is a vertical stack, so its wordmark is unreadable at header
# height. The shield alone is used there, with the name set in HTML beside it;
# the lockup gets the hero, where it has room.
LOCKUP_PATH = ASSETS / "logo.png"
MARK_PATH = ASSETS / "mark.png"


@st.cache_data
def _encode(path_str: str, _mtime: float) -> str:
    """_mtime is part of the cache key, not used in the body: without it the
    cache keeps serving the old bytes after the file on disk changes."""
    return base64.b64encode(pathlib.Path(path_str).read_bytes()).decode()


def _data_uri(path: pathlib.Path) -> str:
    return f"data:image/png;base64,{_encode(str(path), path.stat().st_mtime)}"


def brand_lockup(hero: bool = False) -> str:
    if hero and LOCKUP_PATH.exists():
        return f'<img class="logo" src="{_data_uri(LOCKUP_PATH)}" alt="DanTech Helpdesk"/>'

    mark = (
        f'<img class="mark" src="{_data_uri(MARK_PATH)}" alt=""/>'
        if MARK_PATH.exists()
        else MARK
    )
    return (
        f'{mark}<div class="names"><span class="wordmark">DanTech</span>'
        '<span class="product">IT Helpdesk Agent</span></div>'
    )

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


def landing() -> None:
    """Welcome screen. Runs instead of the chat until we know who is asking.

    Deliberately asymmetric: the message column is wider than the form, and the
    two are offset rather than stacked centre. The conversation view stays on a
    calmer grid, where predictability matters more than composition.
    """
    left, _, right = st.columns([1.15, 0.12, 1], vertical_alignment="center")

    with left:
        st.markdown(
            f'<div class="welcome brandhero">{brand_lockup(hero=True)}'
            '<div class="hero-eyebrow">Tier 1 support</div>'
            "<h1>Describe the problem.<br/>I will take it from there.</h1>"
            "<p>Tell me what is wrong in plain language. I will look up a fix in "
            "our support manuals, or raise a ticket with a human technician if it "
            "needs one.</p></div>",
            unsafe_allow_html=True,
        )

    with right:
        # The card is the form element itself: Streamlit widgets cannot be
        # wrapped in custom HTML, so a hand-rolled card div would render empty
        # with the real fields floating below it.
        with st.form("signin", clear_on_submit=False):
            st.markdown(
                '<div class="formcard-title">Start a ticket</div>',
                unsafe_allow_html=True,
            )
            name = st.text_input("Your name", placeholder="Danishvaran K")
            email = st.text_input("Work email", placeholder="you@company.com")
            submitted = st.form_submit_button("Start a ticket", use_container_width=True)

        if submitted:
            errors = []
            if not name.strip():
                errors.append("Please enter your name.")
            if not EMAIL_PATTERN.match(email.strip()):
                errors.append("Please enter a valid email address.")
            if errors:
                for message in errors:
                    st.error(message)
            else:
                st.session_state.user_name = name.strip()
                st.session_state.user_email = email.strip()
                st.rerun()

        st.markdown(
            '<p class="privacy">Your email is used to send you a copy of any '
            "ticket raised for you, and is stored alongside that ticket in the "
            "queue log. This is a portfolio project running against a synthetic "
            "manual set, not a real support desk.</p>",
            unsafe_allow_html=True,
        )


if "user_email" not in st.session_state:
    landing()
    st.stop()

if "thread" not in st.query_params:
    st.query_params["thread"] = str(uuid.uuid4())
thread_id = st.query_params["thread"]

if st.session_state.get("thread_id") != thread_id:
    st.session_state.thread_id = thread_id
    st.session_state.display_history = [
        {
            "role": "user" if message.type == "human" else "assistant",
            "content": message.content,
            **message.additional_kwargs,
        }
        for message in load_history(thread_id)
    ]

ref = ticket_ref(thread_id)

st.markdown(
    f'<div class="brandbar">{brand_lockup()}'
    f'<span class="ticket">{ref}</span></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    # No monogram here: it already sits in the header a few hundred pixels away.
    st.markdown('<div class="sb-heading first">This ticket</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-item">{ref}<small>Bookmark this page to return to it</small></div>'
        f'<div class="sb-item">{st.session_state.user_name}'
        f'<small>{st.session_state.user_email}</small></div>',
        unsafe_allow_html=True,
    )

    if st.button("Start a new ticket", use_container_width=True):
        st.query_params["thread"] = str(uuid.uuid4())
        st.rerun()

    st.markdown('<div class="sb-heading">What I can help with</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-item">Hardware<small>Laptops, printers, monitors, keyboards</small></div>'
        '<div class="sb-item">Network<small>Wi-Fi, VPN, DNS, connection speed</small></div>'
        '<div class="sb-item">Account<small>Lockouts, MFA, password resets, access</small></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-heading">When I escalate</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-item">Anything affecting many people or a production '
        'system goes straight to the on-call queue. So does anything the '
        'manuals do not confidently cover, rather than me guessing at it.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="lede">Tell me what is broken. I will either walk you through the '
    "fix or get it to someone who can.</p>",
    unsafe_allow_html=True,
)


def render_meta(turn):
    if not turn.get("category"):
        return
    parts = [f'<span class="tag-category">{turn["category"]}</span>']
    sev_class = "tag-critical" if turn.get("severity") == "critical" else ""
    parts.append(f'<span class="{sev_class}">{turn.get("severity", "")}</span>')
    if turn.get("confidence") is not None:
        parts.append(f'<span>match {turn["confidence"]:.2f}</span>')
    st.markdown(f'<div class="meta-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_panel(turn):
    escalated = turn.get("escalated")
    label = "Escalated to a technician" if escalated else "Suggested fix"
    css = "panel escalated" if escalated else "panel"
    st.markdown(
        f'<div class="{css}"><span class="panel-label">{label}</span>{turn["content"]}</div>',
        unsafe_allow_html=True,
    )


for turn in st.session_state.display_history:
    avatar = ":material/person:" if turn["role"] == "user" else ":material/support_agent:"
    with st.chat_message(turn["role"], avatar=avatar):
        if turn["role"] == "user":
            st.write(turn["content"])
        else:
            render_panel(turn)
            render_meta(turn)
            if turn.get("reasoning"):
                with st.expander("Why this classification?"):
                    st.write(turn["reasoning"])

EXAMPLES = [
    "Wi-Fi connects but nothing loads",
    "I'm locked out after too many password attempts",
    "Nobody on the 3rd floor can print",
]

# Resolved before the examples render: st.chat_input pins itself to the bottom
# regardless of call order, and the examples must not still be on screen while
# the message that dismissed them is being answered.
issue = st.chat_input("What is going wrong?") or st.session_state.pop("pending", None)

if not st.session_state.display_history and not issue:
    st.markdown('<div class="empty-label">Common issues</div>', unsafe_allow_html=True)
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"eg{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()

if issue:
    st.session_state.display_history.append({"role": "user", "content": issue})
    with st.chat_message("user", avatar=":material/person:"):
        st.write(issue)

    with st.chat_message("assistant", avatar=":material/support_agent:"):
        with st.spinner("Triaging..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=issue)]},
                config={"configurable": {
                    "thread_id": thread_id,
                    "user_name": st.session_state.user_name,
                    "user_email": st.session_state.user_email,
                }},
            )

        turn = {
            "role": "assistant",
            "content": result["answer"],
            "category": result["category"],
            "severity": result["severity"],
            "confidence": result.get("confidence"),
            "escalated": bool(result.get("escalated")),
            "reasoning": result["reasoning"],
        }
        render_panel(turn)
        render_meta(turn)
        with st.expander("Why this classification?"):
            st.write(turn["reasoning"])

    st.session_state.display_history.append(turn)
