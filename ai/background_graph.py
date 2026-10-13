import logging
import os
from typing import Any, Dict, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

logger = logging.getLogger(__name__)


class BackgroundEvaluatorState(TypedDict, total=False):
    """
    Structured payload state designed specifically for background financial evaluations.
    Maintains system state payloads (user attributes, event triggers, tax limits,
    and generated proactive alert text).
    """
    user_id: Union[int, str]
    user_name: Optional[str]
    financial_event: str
    remaining_80c_limit: float
    additional_context: Optional[Dict[str, Any]]
    alert_message: str


# Safe API key resolution ensuring Pydantic does not crash during import/collection in CI
api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or "dummy_test_key_so_pydantic_does_not_crash"
)

# Initialize Gemini 3.5 Pro for headless financial reasoning
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-pro",
    temperature=0.2,
    google_api_key=api_key,
)


async def evaluate_financial_event_node(state: BackgroundEvaluatorState) -> Dict[str, Any]:
    """
    LangGraph node that consumes the structured financial state and pending 80C tax deduction limit,
    prompting Gemini 3.5 Pro to synthesize a concise, personalized, and actionable proactive alert.
    """
    user_id = state.get("user_id", "Unknown")
    user_name = state.get("user_name") or f"User {user_id}"
    financial_event = state.get(
        "financial_event",
        "Financial year-end approaching, user has ₹50k remaining in 80C"
    )
    remaining_80c = state.get("remaining_80c_limit", 50000.0)

    system_prompt = (
        "You are ChronoFi's Proactive Autonomous Financial Advisor specializing in Indian tax optimization. "
        "Your role is to analyze a user's background financial state and pending tax limits to generate an urgent, "
        "highly actionable, and personalized alert.\n"
        "Rules:\n"
        "1. Keep the message concise (2 to 3 sentences maximum).\n"
        "2. Recommend concrete Section 80C investment options (e.g., ELSS mutual funds, PPF, or Tax-Saver FDs).\n"
        "3. Highlight potential tax savings or upcoming deadlines.\n"
        "4. Tone must be professional, direct, and actionable."
    )

    user_prompt = (
        f"User Details:\n"
        f"- ID: {user_id}\n"
        f"- Name: {user_name}\n"
        f"- Financial Event: {financial_event}\n"
        f"- Unutilized Section 80C Limit: ₹{remaining_80c:,.2f}\n\n"
        "Generate a short, actionable, personalized tax deduction alert message."
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        alert_message = response.content.strip() if response and response.content else ""
    except Exception as exc:
        logger.warning(
            "Gemini 3.5 Pro model invocation encountered an issue (%s). Using deterministic fallback.",
            exc,
        )
        # Resilient fallback ensuring seamless alerts in local dev, testing, or rate-limited environments
        approx_tax_saved = int(remaining_80c * 0.312)
        alert_message = (
            f"Tax Alert for {user_name}: Financial year-end is approaching with ₹{remaining_80c:,.0f} "
            f"unclaimed under Section 80C. Invest in ELSS tax-saving funds or PPF before March 31st "
            f"to save up to ₹{approx_tax_saved:,} in taxes!"
        )

    return {"alert_message": alert_message}


# Define the headless StateGraph
workflow = StateGraph(BackgroundEvaluatorState)

# Register the evaluation node
workflow.add_node("evaluate_financial_event", evaluate_financial_event_node)

# Linear headless transition from START -> evaluate_financial_event -> END
workflow.add_edge(START, "evaluate_financial_event")
workflow.add_edge("evaluate_financial_event", END)

# Compile the headless graph for programmatic execution
background_evaluator_graph = workflow.compile()

# Public aliases for convenient importing
background_graph = background_evaluator_graph
evaluator_graph = background_evaluator_graph
