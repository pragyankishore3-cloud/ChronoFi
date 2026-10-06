# ChronoFi: The Problem

## Original Problem Statement
Retail investors and high-net-worth individuals lack access to institutional-grade quantitative forecasting tools. Traditional financial planning relies on deterministic, linear calculators (e.g., "assuming 8% compounding interest"). This creates a false sense of security, ignoring market volatility, tail-risk (Black Swan events), and sequence-of-returns risk.

## Target Users
- Ambitious young professionals (e.g., software engineers, doctors).
- High-Net-Worth Individuals (HNWIs) planning early retirement (FIRE).

## Core Problem
Users do not know the *true probability* of achieving their financial goals given the chaotic nature of the stock market.

## Desired Outcome
Provide an AI-driven terminal that uses deep learning (LSTMs) to run thousands of Monte Carlo Dropout simulations on historical market data, returning probabilistic outcomes (e.g., "You have a 72% chance of hitting your target").

## Explicit Constraints
- Do not provide direct, actionable investment advice (e.g., "Buy TSLA").
- Must use rigorous mathematical modeling, not LLM hallucinations for math.
- The UI must remain clean, professional, and institutional (no generic "AI aesthetics").

## Out of Scope
- Direct brokerage integration / trade execution.
- Real-time stock picking algorithms.
