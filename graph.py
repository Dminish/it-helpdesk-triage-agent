"""LangGraph IT helpdesk triage agent.

classify -> escalate (critical)
         -> retrieve -> escalate (low-confidence match)
                      -> answer

Conversation memory via a MemorySaver checkpointer keyed by thread_id.
"""
import csv
import os
import sqlite3
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pinecone import Pinecone
from pydantic import BaseModel, Field

ESCALATION_LOG = "escalations.csv"
CHECKPOINT_DB = "checkpoints.sqlite"
INDEX_NAME = os.environ.get("PINECONE_INDEX", "it-helpdesk-manuals")
# Calibrated by eval_retrieval.py: answerable queries scored >= 0.459, unanswerable
# ones <= 0.418, so this sits in the middle of that gap. Re-run that script after
# changing the embedding model or the manual corpus; the gap is only 0.041 wide.
CONFIDENCE_THRESHOLD = 0.44

# Local Ollama by default; hosted OpenAI when deployed, where there is no GPU to
# run a local model on. Embeddings always come from OpenAI, so the retrieval
# threshold above holds either way -- but the classifier numbers in the README
# were measured on qwen2.5, so re-run eval_classifier.py if you switch.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

if LLM_PROVIDER == "openai":
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
elif LLM_PROVIDER == "ollama":
    llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "qwen2.5"), temperature=0)
else:
    raise ValueError(f"LLM_PROVIDER must be 'ollama' or 'openai', got {LLM_PROVIDER!r}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pinecone_index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)


def search_manuals(query: str, category: str, k: int = 3) -> tuple[list[str], float]:
    vec = embeddings.embed_query(query)
    result = pinecone_index.query(
        vector=vec, top_k=k, filter={"category": category}, include_metadata=True
    )
    matches = result["matches"]
    snippets = [m["metadata"]["text"] for m in matches]
    top_score = matches[0]["score"] if matches else 0.0
    return snippets, top_score


class Triage(BaseModel):
    category: Literal["Hardware", "Network", "Account"] = Field(
        description="Best-fit category for the issue"
    )
    severity: Literal["critical", "normal"] = Field(
        description=(
            "'critical' only for outages/security incidents affecting many "
            "users or production systems (e.g. server down, data breach, "
            "site-wide network outage). Everything else is 'normal'."
        )
    )
    reasoning: str = Field(description="One sentence explaining the classification")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    category: str
    severity: str
    reasoning: str
    confidence: float | None
    context: str
    answer: str
    escalated: bool


classify_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Tier 1 IT helpdesk triage system. Classify the ticket into "
        "a category and severity.\n\n"
        "severity='critical' is ONLY for incidents affecting many users or "
        "production systems: a server/site outage, a security breach, or "
        "something stopping an entire team from working. It is NOT about how "
        "urgent the issue feels to one person.\n"
        "severity='normal' is everything else, including issues that are "
        "very annoying or block one person's work, like a broken keyboard, a "
        "locked account, a VPN timeout. A single person being unable to work "
        "is still 'normal', not 'critical'.\n"
        "Scope is often implied rather than stated. Naming a team, department, "
        "floor, office, or site means many users are affected, even without a "
        "word like 'entire' or 'everyone'. 'Network down in the marketing "
        "department' is critical for the same reason 'entire marketing "
        "department cannot connect' is.\n\n"
        "Examples:\n"
        "- 'my keyboard keys are sticking' -> Hardware, normal\n"
        "- 'locked out of my account after failed logins' -> Account, normal\n"
        "- 'vpn keeps timing out' -> Network, normal\n"
        "- 'the production server room caught fire' -> Hardware, critical\n"
        "- 'prod database is completely down for everyone' -> Network, critical\n\n"
        "You may be shown earlier turns of the conversation. Classify the "
        "user's LATEST message, using the earlier turns only as context. A "
        "follow-up like 'that didn't work' inherits the original issue's "
        "category; it does not become critical just because it is a repeat."
    )),
    ("placeholder", "{messages}"),
])
classifier = classify_prompt | llm.with_structured_output(Triage)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a Tier 1 IT support technician chatting with a user across "
        "multiple turns. Using ONLY the manual excerpts below, give clear "
        "step-by-step troubleshooting for their latest message. If they say "
        "a previous suggestion didn't work, acknowledge that and try the next "
        "step instead of repeating yourself.\n\n"
        "Manual excerpts:\n{context}"
    )),
    ("placeholder", "{messages}"),
])
answerer = answer_prompt | llm


CLASSIFY_WINDOW = 5  # recent messages fed to the classifier for context


def classify(state: AgentState) -> dict:
    result = classifier.invoke({"messages": state["messages"][-CLASSIFY_WINDOW:]})
    return {
        "category": result.category,
        "severity": result.severity,
        "reasoning": result.reasoning,
        "confidence": None,  # clear stale value from a prior turn's retrieval
    }


def route_after_classify(state: AgentState) -> Literal["escalate", "retrieve"]:
    return "escalate" if state["severity"] == "critical" else "retrieve"


RETRIEVAL_TURNS = 3  # recent user messages combined into the retrieval query


def retrieval_query(messages: list[BaseMessage]) -> str:
    """Recent user turns joined into one query.

    A follow-up like "i tried that, still failing" carries no topic words of its
    own, so embedding it alone retrieves nothing and the low score escalates a
    question the manuals actually answer. Assistant replies are left out: they
    already quote the manuals, so including them would bias retrieval toward
    whatever was answered last.
    """
    user_turns = [m.content for m in messages if m.type == "human"]
    return " ".join(user_turns[-RETRIEVAL_TURNS:])


def retrieve(state: AgentState) -> dict:
    query = retrieval_query(state["messages"])
    snippets, confidence = search_manuals(query, state["category"])
    context = "\n\n".join(snippets) or "No matching manual found."
    return {"confidence": confidence, "context": context}


def route_after_retrieve(state: AgentState) -> Literal["escalate", "answer"]:
    return "escalate" if state["confidence"] < CONFIDENCE_THRESHOLD else "answer"


def turn_meta(state: AgentState, escalated: bool) -> dict:
    """Metadata carried on the reply itself, so a restored conversation still
    shows the tags that were displayed when it was first answered."""
    return {
        "category": state["category"],
        "severity": state["severity"],
        "reasoning": state["reasoning"],
        "confidence": state.get("confidence"),
        "escalated": escalated,
    }


def escalate(state: AgentState) -> dict:
    if state["severity"] == "critical":
        trigger = "critical severity"
        answer_text = (
            f"This looks critical ({state['category']}), escalated to the "
            "human on-call queue. A technician will follow up shortly."
        )
    else:
        trigger = "no confident manual match"
        answer_text = (
            f"I couldn't find a confident match in the manuals for this "
            f"{state['category']} issue, escalated to a human technician "
            "rather than guess."
        )

    is_new = not os.path.exists(ESCALATION_LOG)
    with open(ESCALATION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "category", "trigger", "issue", "reasoning"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            state["category"],
            trigger,
            state["messages"][-1].content,
            state["reasoning"],
        ])
    return {
        "escalated": True,
        "answer": answer_text,
        "messages": [
            AIMessage(content=answer_text, additional_kwargs=turn_meta(state, True))
        ],
    }


def answer(state: AgentState) -> dict:
    result = answerer.invoke({"context": state["context"], "messages": state["messages"]})
    result.additional_kwargs.update(turn_meta(state, False))
    return {"answer": result.content, "escalated": False, "messages": [result]}


builder = StateGraph(AgentState)
builder.add_node("classify", classify)
builder.add_node("retrieve", retrieve)
builder.add_node("escalate", escalate)
builder.add_node("answer", answer)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify", route_after_classify, {"escalate": "escalate", "retrieve": "retrieve"}
)
builder.add_conditional_edges(
    "retrieve", route_after_retrieve, {"escalate": "escalate", "answer": "answer"}
)
builder.add_edge("escalate", END)
builder.add_edge("answer", END)

# check_same_thread=False: Streamlit reruns the script on a different thread than
# the one that opened the connection.
checkpointer = SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)


def load_history(thread_id: str) -> list[BaseMessage]:
    """Messages already stored for this thread, oldest first. Empty if new."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values.get("messages", []) if snapshot.values else []
