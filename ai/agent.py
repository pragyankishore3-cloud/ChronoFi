from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from engine.monte_carlo import run_monte_carlo_sip
from engine.math_engine import run_scenario
import os
import asyncio

@tool
def simulate_goal_probability_tool(target_amount: float, current_savings: float, monthly_sip: float, years: int) -> str:
    """
    Runs a Monte Carlo simulation (10,000 market paths) to determine the exact probability of a user hitting their financial goal given a specific monthly savings amount.
    Use this tool to test if a user's proposed savings plan is realistic.
    """
    result = run_monte_carlo_sip(target_amount, current_savings, monthly_sip, years)
    
    return (
        f"Probability of hitting target (₹{target_amount}): {result['probability_of_success_percent']}%\n"
        f"Worst Case (10th Percentile Market): ₹{result['worst_case_10th_percentile']}\n"
        f"Expected Case (50th Percentile Market): ₹{result['expected_50th_percentile']}\n"
        f"Best Case (90th Percentile Market): ₹{result['best_case_90th_percentile']}"
    )

@tool
def run_scenario_tool(base_salary: float, monthly_expenses: float, months_unemployed: int) -> str:
    """Calculates the basic financial impact of a life event, like taking a sabbatical."""
    result = run_scenario(base_salary, monthly_expenses, months_unemployed)
    return f"Annual Surplus/Deficit: ₹{result['annual_surplus']}. Is Deficit: {result['is_deficit']}"

tools = [simulate_goal_probability_tool, run_scenario_tool]

system_message = """You are Chronofi, a highly professional financial AI co-pilot.
Your job is to help users understand the true probability of achieving their financial goals using Monte Carlo simulations.

CRITICAL RULES:
1. You MUST NOT perform math yourself. Always use `simulate_goal_probability_tool` to test if their SIP is enough.
2. When explaining results, clearly translate the percentiles: 
   - Explain the "Expected Case" as average market conditions.
   - Explain the "Worst Case" as what happens if the market underperforms.
3. If probability of success is below 80%, gently suggest they either increase their SIP, delay the target date, or lower the target amount.
"""

def get_financial_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    agent = create_react_agent(llm, tools, state_modifier=system_message)
    return agent
