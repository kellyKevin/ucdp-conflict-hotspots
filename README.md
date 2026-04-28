# Conflict Pattern Analysis Dashboard

This project provides a comprehensive data science toolkit for analyzing conflict patterns using the UCDP Georeferenced Event Dataset (GED) and World Bank economic indicators.

## Features

- **Data Cleaning & Prep**: Automated processing of UCDP GED event data.
- **Exploratory Data Analysis**: Temporal and geographic visualization of conflict frequency and fatalities.
- **Hot Zone Detection**: Spatial clustering using DBSCAN to identify high-intensity conflict zones.
- **Risk Forecasting**: Time-series prediction of conflict escalation using Prophet.
- **Economic Correlation**: Analysis of how conflict intensity correlates with GDP growth and Inflation (World Bank data).
- **AI Decision Simulation**: A Reinforcement Learning (PPO) agent that simulates company supply chain decisions (Route, Reroute, Halt) under varying conflict risks.
- **Interactive Dashboard**: A Streamlit-based interface to explore all the above features.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Files

- `dashboard.py`: The interactive Streamlit application.
- `data_prep.py`: Script for loading and cleaning the UCDP GED dataset.
- `clustering.py`: Implementation of DBSCAN clustering for hotspot detection.
- `forecasting.py`: Prophet-based time-series forecasting.
- `economic_analysis.py`: Correlation analysis with World Bank data.
- `rl_env.py`: Custom Gymnasium environment for the supply chain simulation.
- `train_rl.py`: Script to train the Reinforcement Learning agent.
- `requirements.txt`: List of required Python packages.

## Data Source

- **Conflict Data**: UCDP Georeferenced Event Dataset (GED) Global version 25.1.
- **Economic Data**: World Bank (via `wbgapi`).
