import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from graph import graph, load_history

st.set_page_config(page_title="Triage Agent", page_icon=":material/terminal:", layout="centered")

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
.block-container { max-width: 720px; padding-top: 3.5rem; }

.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
h1.title {
  font-size: 2rem;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.01em;
}
p.subtitle {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.55;
  max-width: 58ch;
  margin-bottom: 2rem;
}

[data-testid="stChatMessage"] {
  background: transparent;
  border: none;
  padding: 0.4rem 0;
}
[data-testid="stChatMessageContent"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1.1rem;
}
[data-testid="stChatMessageAvatarUser"] {
  background: var(--surface-2) !important;
  color: var(--text) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--accent-dim) !important;
  color: var(--accent) !important;
}

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

[data-testid="stChatInput"] {
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 8px;
  background: var(--surface-2);
  box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stChatInputSubmitButton"] {
  background: var(--accent) !important;
  border-radius: 6px !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: var(--accent-ink) !important; }

.streamlit-expanderHeader {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  color: var(--muted);
}
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown('<div class="eyebrow">TIER 1 · HELPDESK</div>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Triage Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Describe the issue in plain language. It gets classified, then either '
    "routed to a fix pulled from the manuals or escalated to a human. The conversation is "
    "remembered, so follow-ups build on what came before.</p>",
    unsafe_allow_html=True,
)

# The thread id lives in the URL, not session_state: session_state dies with the
# server, which would orphan the stored conversation it is meant to reopen.
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


def render_meta(turn):
    if not turn.get("category"):
        return
    parts = [f'<span class="tag-category">{turn["category"]}</span>']
    sev_class = "tag-critical" if turn.get("severity") == "critical" else ""
    parts.append(f'<span class="{sev_class}">{turn.get("severity", "")}</span>')
    if turn.get("confidence") is not None:
        parts.append(f'<span>confidence {turn["confidence"]:.2f}</span>')
    st.markdown(f'<div class="meta-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_panel(turn):
    if turn.get("escalated"):
        st.markdown(
            f'<div class="panel escalated"><span class="panel-label">Escalated</span>{turn["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="panel"><span class="panel-label">Suggested fix</span>{turn["content"]}</div>',
            unsafe_allow_html=True,
        )


for turn in st.session_state.display_history:
    with st.chat_message(turn["role"], avatar=(":material/person:" if turn["role"] == "user" else ":material/terminal:")):
        if turn["role"] == "user":
            st.write(turn["content"])
        else:
            render_panel(turn)
            render_meta(turn)
            if turn.get("reasoning"):
                with st.expander("Why this classification?"):
                    st.write(turn["reasoning"])

issue = st.chat_input("my laptop wont turn on and i have a meeting in 5 min")

if issue:
    st.session_state.display_history.append({"role": "user", "content": issue})
    with st.chat_message("user", avatar=":material/person:"):
        st.write(issue)

    with st.chat_message("assistant", avatar=":material/terminal:"):
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
