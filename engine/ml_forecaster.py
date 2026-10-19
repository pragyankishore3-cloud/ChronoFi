import os
import pickle

import numpy as np

try:
    import torch
    from ml.model import ChronoFiLSTM
    HAS_TORCH = True
except (ImportError, Exception):
    HAS_TORCH = False
    torch = None
    ChronoFiLSTM = None

# Load the model and scaler lazily
_model = None
_scaler = None

def load_ml_resources():
    global _model, _scaler
    if not HAS_TORCH:
        return
    base_dir = os.path.join(os.path.dirname(__file__), "..", "ml")
    
    if _scaler is None:
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                _scaler = pickle.load(f)
            
    if _model is None:
        model_path = os.path.join(base_dir, "chronofi_lstm.pth")
        if os.path.exists(model_path):
            _model = ChronoFiLSTM(input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2)
            _model.load_state_dict(torch.load(model_path, weights_only=True))
            # Keep the model in training mode to enable Monte Carlo Dropout during inference
            _model.train() 

def run_ml_forecaster(
    target_amount: float,
    current_savings: float,
    monthly_sip: float,
    years: int,
    initial_equity_allocation: float = 0.80,
    num_simulations: int = 1000
) -> dict:
    """
    Runs an LSTM inference engine when PyTorch is available,
    falling back to vectorized simulation with glide path when PyTorch is omitted.
    """
    load_ml_resources()
    months = years * 12

    if not HAS_TORCH or _model is None or _scaler is None:
        # High-performance Monte Carlo fallback
        mean_monthly_return = 0.12 / 12
        monthly_volatility = 0.15 / np.sqrt(12)
        random_returns = np.random.normal(mean_monthly_return, monthly_volatility, (months, num_simulations))
        growth_multipliers = 1 + random_returns
        portfolio_paths = np.zeros((months + 1, num_simulations))
        portfolio_paths[0] = current_savings

        for month in range(1, months + 1):
            portfolio_paths[month] = (portfolio_paths[month - 1] * growth_multipliers[month - 1]) + monthly_sip

        gross_final_values = portfolio_paths[-1]
        total_principal = current_savings + (monthly_sip * months)
        capital_gains = np.maximum(0, gross_final_values - total_principal)
        final_values = gross_final_values - (capital_gains * 0.10)
        inflated_target = target_amount * ((1 + 0.06) ** years)

        successes = np.sum(final_values >= inflated_target)
        prob_success = (successes / num_simulations) * 100
        worst_case = np.percentile(final_values, 10)
        expected = np.percentile(final_values, 50)
        best_case = np.percentile(final_values, 90)
        worst_10_percent_mask = final_values <= worst_case
        cvar = np.mean(final_values[worst_10_percent_mask])

        return {
            "probability_of_success_percent": round(float(prob_success), 2),
            "median_inflated_target": round(float(inflated_target), 2),
            "expected_50th_percentile": round(float(expected), 2),
            "worst_case_10th_percentile": round(float(worst_case), 2),
            "best_case_90th_percentile": round(float(best_case), 2),
            "conditional_value_at_risk_cvar": round(float(cvar), 2),
            "total_principal_invested": round(float(total_principal), 2)
        }
    
    neutral_scaled_return = _scaler.transform([[0.0]])[0][0]
    initial_seq = torch.full((num_simulations, 12, 1), neutral_scaled_return, dtype=torch.float32)
    
    # Store the simulated paths
    portfolio_paths = np.zeros((months + 1, num_simulations))
    portfolio_paths[0] = current_savings
    
    # Fixed debt return (4% annual)
    debt_annual_return = 0.04
    debt_monthly_return = debt_annual_return / 12
    
    current_seq = initial_seq.clone()
    
    # Pre-calculate glide path
    final_equity_allocation = 0.40
    equity_allocations = np.linspace(initial_equity_allocation, final_equity_allocation, months)
    
    with torch.no_grad(): # Don't track gradients, but dropout will still run because of .train()
        for m in range(1, months + 1):
            # Predict the next scaled return
            scaled_pred = _model(current_seq) # shape: (num_simulations, 1)
            
            # Inverse transform to get actual returns
            # Move to CPU numpy, inverse transform, then move back
            scaled_np = scaled_pred.numpy()
            actual_returns = _scaler.inverse_transform(scaled_np).flatten()
            
            current_eq_alloc = equity_allocations[m - 1]
            current_debt_alloc = 1.0 - current_eq_alloc
            
            blended_returns = (current_eq_alloc * actual_returns) + (current_debt_alloc * debt_monthly_return)
            
            portfolio_paths[m] = (portfolio_paths[m - 1] * (1 + blended_returns)) + monthly_sip
            
            # Slide the window
            scaled_pred_reshaped = scaled_pred.unsqueeze(1) # shape: (num_simulations, 1, 1)
            current_seq = torch.cat((current_seq[:, 1:, :], scaled_pred_reshaped), dim=1)

    # Calculate statistics
    gross_final_values = portfolio_paths[-1]
    total_principal = current_savings + (monthly_sip * months)
    capital_gains = np.maximum(0, gross_final_values - total_principal)
    final_values = gross_final_values - (capital_gains * 0.10) # 10% LTCG
    
    # Target inflation (naive 6% for the ML engine to keep it focused on portfolio ML)
    inflated_target = target_amount * ((1 + 0.06) ** years)
    
    successes = np.sum(final_values >= inflated_target)
    prob_success = (successes / num_simulations) * 100
    
    worst_case = np.percentile(final_values, 10)
    expected = np.percentile(final_values, 50)
    best_case = np.percentile(final_values, 90)
    
    # CVaR (Average of the worst 10%)
    worst_10_percent_mask = final_values <= worst_case
    cvar = np.mean(final_values[worst_10_percent_mask])
    
    return {
        "probability_of_success_percent": round(prob_success, 2),
        "median_inflated_target": round(inflated_target, 2),
        "expected_50th_percentile": round(expected, 2),
        "worst_case_10th_percentile": round(worst_case, 2),
        "best_case_90th_percentile": round(best_case, 2),
        "conditional_value_at_risk_cvar": round(cvar, 2),
        "total_principal_invested": round(total_principal, 2)
    }
