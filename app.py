import os
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

:root {
  --bg: #131417;
  --surface: #1E2025;
  --surface-2: #272A31;
  --border: rgba(255,255,255,0.14);
  --text: #ECEAE4;
  --muted: #93969E;
  --accent: #FFB454;
  --accent-ink: #1A1409;
  --critical: #FF6B6B;
  --critical-dim: rgba(255,107,107,0.14);
  --accent-dim: rgba(255,180,84,0.14);
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
.brandbar .mark { flex: 0 0 auto; }
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
  border: 1px solid rgba(255,180,84,0.35);
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
  background: rgba(255,255,255,0.08);
  color: var(--text);
  padding: 0.1em 0.35em;
  border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.88em;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #101114; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .sb-heading {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.4rem 0 0.5rem 0;
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
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 8px;
  background: var(--surface-2);
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 6px !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--accent-ink) !important; }

[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# Wordmark for an invented brand: a simple geometric monogram beats a text-only
# logo, and inline SVG keeps it theme-aware without an asset request.
# Kept on one line: st.markdown renders indented HTML as a code block.
MARK = (
    '<svg class="mark" width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">'
    '<rect x="0.75" y="0.75" width="32.5" height="32.5" rx="8" fill="rgba(255,180,84,0.12)" '
    'stroke="rgba(255,180,84,0.45)" stroke-width="1.5"/>'
    '<path d="M11 10.5h5.2c3.9 0 6.3 2.5 6.3 6.5s-2.4 6.5-6.3 6.5H11V10.5z" stroke="#FFB454" '
    'stroke-width="2.1" stroke-linejoin="round" fill="none"/>'
    '<circle cx="23.4" cy="11.4" r="2.1" fill="#FFB454"/>'
    "</svg>"
)

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
    f'<div class="brandbar">{MARK}'
    f'<div class="names"><span class="wordmark">DanTech</span>'
    f'<span class="product">IT Helpdesk Agent</span></div>'
    f'<span class="ticket">{ref}</span></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"{MARK}", unsafe_allow_html=True)
    st.markdown('<div class="sb-heading">This ticket</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-item">{ref}<small>Bookmark this page to return to it</small></div>', unsafe_allow_html=True)

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
    '<p class="lede">Describe your issue in plain language. I will classify it, '
    "look up a fix in the support manuals, and hand it to a human technician if "
    "it is urgent or outside what the manuals cover. Follow-up questions stay on "
    "the same ticket.</p>",
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

issue = st.chat_input("Describe your IT issue")

if issue:
    st.session_state.display_history.append({"role": "user", "content": issue})
    with st.chat_message("user", avatar=":material/person:"):
        st.write(issue)

    with st.chat_message("assistant", avatar=":material/support_agent:"):
        with st.spinner("Triaging..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=issue)]},
                config={"configurable": {"thread_id": thread_id}},
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
