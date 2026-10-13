
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from ai.tools import calculate_tax_liability
from engine.math_engine import run_scenario
from engine.ml_forecaster import run_ml_forecaster


@tool
def simulate_goal_probability_tool(target_amount: float, current_savings: float, monthly_sip: float, years: int) -> str:
    """
    Runs a probabilistic machine learning forecast using a PyTorch LSTM trained on historical S&P 500 data.
    Uses Monte Carlo Dropout to generate 500 potential market trajectories.
    Use this to determine the exact probability of hitting a financial goal.
    """
    result = run_ml_forecaster(
        target_amount, 
        current_savings, 
        monthly_sip, 
        years, 
        num_simulations=500
    )
    
    return (
        f"LSTM ML Forecast Probability of hitting target: {result['probability_of_success_percent']}%\n"
        f"Inflated Target Amount (Median): ₹{result['median_inflated_target']}\n"
        f"Expected Portfolio Value (50th Percentile): ₹{result['expected_50th_percentile']}\n"
        f"Worst Case Scenario (10th Percentile): ₹{result['worst_case_10th_percentile']}\n"
        f"CVaR (Average of worst 10% market crashes): ₹{result['conditional_value_at_risk_cvar']}\n"
        f"Best Case Scenario (90th Percentile): ₹{result['best_case_90th_percentile']}"
    )

@tool
def run_scenario_tool(base_salary: float, monthly_expenses: float, months_unemployed: int) -> str:
    """Calculates the basic financial impact of a life event, like taking a sabbatical."""
    result = run_scenario(base_salary, monthly_expenses, months_unemployed)
    return f"Annual Surplus/Deficit: ₹{result['annual_surplus']}. Is Deficit: {result['is_deficit']}"

tools = [simulate_goal_probability_tool, run_scenario_tool, calculate_tax_liability]

system_message = """You are Chronofi, a highly professional financial AI co-pilot and quantitative analyst.
Your job is to help users understand the true probability of achieving their financial goals using our institutional-grade Monte Carlo engine.

CRITICAL RULES:
1. You MUST NOT perform math yourself. Always use `simulate_goal_probability_tool` to test if their SIP is enough.
2. When explaining results, clearly translate the quantitative metrics:
   - Explain the "Inflated Target" as what the goal will actually cost in the future due to cost-of-living increases.
   - Explain the "Expected Case" as average market conditions (50th percentile).
   - Explain "CVaR" (Conditional Value at Risk) as the severe "Black Swan" crash scenario—what happens in the absolute worst 10% of market crashes.
3. If probability of success is below 80%, gently suggest they either increase their SIP, delay the target date, or lower the target amount. Do not offer specific asset advice (e.g., "buy this stock"), speak only of broad asset classes.
"""

def get_financial_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    agent = create_react_agent(llm, tools, prompt=system_message)
    return agent
