from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from engine.math_engine import calculate_future_value, calculate_required_sip, run_scenario
import os

# Create tools from the math engine functions
@tool
def calculate_sip_tool(target_amount: float, current_savings: float, expected_annual_return: float, years: int) -> str:
    """
    Calculates the required monthly savings (SIP) to hit a financial goal. 
    Use this tool whenever a user asks if they can afford something, or how much they need to save per month.
    """
    result = calculate_required_sip(target_amount, current_savings, expected_annual_return, years)
    return f"Required Monthly SIP: ₹{result['required_monthly_sip']}. (Inflation adjusted target cost: ₹{result['inflation_adjusted_target']})"

@tool
def run_scenario_tool(base_salary: float, monthly_expenses: float, months_unemployed: int) -> str:
    """
    Calculates the financial impact of a life event, like taking a sabbatical or losing a job.
    """
    result = run_scenario(base_salary, monthly_expenses, months_unemployed)
    return f"Annual Surplus/Deficit: ₹{result['annual_surplus']}. Is Deficit: {result['is_deficit']}"

tools = [calculate_sip_tool, run_scenario_tool]

system_message = """You are Chronofi, a highly professional yet approachable financial AI co-pilot.
Your job is to help users understand if they can achieve their financial goals.

CRITICAL RULES:
1. You MUST NOT perform any financial calculations yourself. You MUST use the provided tools for any math.
2. Maintain a professional, advisor-like tone but explain things simply without dense jargon.
3. If a goal requires a higher monthly saving than the user's surplus, gently explain the trade-offs.
4. Do not offer specific stock or mutual fund advice. Speak in terms of 'expected return rates' and 'asset classes'.
"""

def get_financial_agent():
    # Make sure GEMINI_API_KEY is in the environment
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    # Using modern LangGraph prebuilt agent
    agent = create_react_agent(llm, tools, state_modifier=system_message)
    return agent
