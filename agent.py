
import re
from typing import Callable, TypedDict, Any

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "llama3.2:3b"

INTENTS = [
    "document_qa",
    "study_plan",
    "quiz",
    "general",
]


# =========================================================
# AGENT STATE
# =========================================================

class AgentState(TypedDict, total=False):
    user_input: str
    intent: str
    selected_tool: str
    response: str


# =========================================================
# INTENT CLASSIFICATION
# =========================================================

def keyword_fallback(user_input: str) -> str:
    """
    Lightweight fallback if Ollama cannot classify the request.
    """
    text = user_input.lower()

    quiz_words = [
        "quiz",
        "mcq",
        "multiple choice",
        "test me",
        "practice questions",
        "test my knowledge",
    ]

    plan_words = [
        "study plan",
        "learning plan",
        "roadmap",
        "learning path",
        "study schedule",
    ]

    general_words = [
        "hello",
        "hi",
        "good morning",
        "good evening",
        "thank you",
        "thanks",
    ]

    if any(word in text for word in quiz_words):
        return "quiz"

    if any(word in text for word in plan_words):
        return "study_plan"

    if any(text.strip().startswith(word) for word in general_words):
        return "general"

    return "document_qa"



def detect_intent_by_rules(user_input: str):
    """Identify clear requests before asking the LLM to classify them."""
    text = user_input.lower()

    # Check quiz requests first to handle requests like
    # "Create a study plan and quiz me on it."
    quiz_phrases = [
        "quiz",
        "mcq",
        "multiple choice",
        "test me",
        "test my knowledge",
        "practice questions",
        "take a test",
    ]

    if any(phrase in text for phrase in quiz_phrases):
        return "quiz"

    study_plan_phrases = [
        "study plan",
        "learning plan",
        "roadmap",
        "learning path",
        "study schedule",
        "plan to learn",
        "plan for learning",
    ]

    if any(phrase in text for phrase in study_plan_phrases):
        return "study_plan"

    general_phrases = [
        "hello",
        "hi",
        "good morning",
        "good evening",
        "thank you",
        "thanks",
    ]

    if text.strip() in general_phrases:
        return "general"

    # None means the request needs further classification.
    return None


def classify_intent(user_input: str) -> str:
    """Route clear requests using rules; classify ambiguous ones locally."""

    # 1. Handle recognizable requests without an LLM call.
    rule_intent = detect_intent_by_rules(user_input)

    if rule_intent:
        return rule_intent

    # 2. Ask the local LLM about ambiguous requests.
    router = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

    prompt = f"""
You route requests for an AI Learning Assistant.

Choose exactly one intent:

study_plan: The user requests a study plan, learning roadmap,
or structured learning schedule.

quiz: The user requests a quiz, MCQs, or a knowledge test.

document_qa: The user asks a technical question about AI,
ML, DL, Generative AI, RAG, AWS, or learning documents.

general: Greetings, thanks, or casual conversation.

Return exactly one label:
study_plan
quiz
document_qa
general

USER REQUEST:
{user_input}

INTENT:
"""

    try:
        result = router.invoke(prompt)
        output = str(result.content).strip().lower()

        # Prefer an exact label rather than matching incidental text.
        output = output.strip("`'\" .\n")

        if output in INTENTS:
            return output

        # Accept a label on the first line if the model adds
        # a brief explanation afterward.
        first_line = output.splitlines()[0].strip("`'\" .")

        if first_line in INTENTS:
            return first_line

    except Exception as error:
        print(f"LLM routing failed; using fallback: {error}")

    return keyword_fallback(user_input)


# =========================================================
# TOOL EXECUTION NODES
# =========================================================

def make_tool_node(
    intent: str,
    tool_functions: dict[str, Callable[[str], Any]],
):
    """
    Create a graph node that invokes the selected Python tool.
    """

    def execute_tool(state: AgentState) -> dict:
        user_input = state.get("user_input", "").strip()

        try:
            tool = tool_functions[intent]
            result = tool(user_input)

            return {
                "selected_tool": intent,
                "response": str(result),
            }

        except Exception as error:
            return {
                "selected_tool": intent,
                "response": (
                    f"The {intent} tool encountered an error: {error}"
                ),
            }

    return execute_tool


# =========================================================
# LANGGRAPH AGENT
# =========================================================

def build_agent(
    tool_functions: dict[str, Callable[[str], Any]],
):
    """
    Build a LangGraph agent that routes the user's request
    to one of four registered tools.

    Expected tool keys:
    document_qa, study_plan, quiz, general
    """

    missing_tools = [
        intent
        for intent in INTENTS
        if intent not in tool_functions
    ]

    if missing_tools:
        raise ValueError(
            f"Missing tool implementations: {missing_tools}"
        )

    workflow = StateGraph(AgentState)

    # First node: understand and classify the request.
    def router_node(state: AgentState) -> dict:
        user_input = state.get("user_input", "")

        intent = classify_intent(user_input)

        print(f"Agent selected intent: {intent}")

        return {"intent": intent}

    workflow.add_node("router", router_node)

    # Tool nodes.
    for intent in INTENTS:
        workflow.add_node(
            intent,
            make_tool_node(intent, tool_functions),
        )

    workflow.set_entry_point("router")

    # Route from the classifier to the chosen tool.
    workflow.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "document_qa": "document_qa",
            "study_plan": "study_plan",
            "quiz": "quiz",
            "general": "general",
        },
    )

    # Return the tool response to the caller.
    for intent in INTENTS:
        workflow.add_edge(intent, END)

    return workflow.compile()
