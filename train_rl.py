import pandas as pd
import numpy as np
from rl_env import ConflictSupplyChainEnv
from stable_baselines3 import PPO

def train_rl_agent():
    df = pd.read_pickle('cleaned_data.pkl')
    country_df = df[df['country'] == 'Afghanistan'].copy()
    country_df['ds'] = country_df['date_end'].dt.to_period('M').dt.to_timestamp()
    ts = country_df.groupby('ds')['best'].sum()
    risk_series = ts / (ts.max() + 1e-6)

    env = ConflictSupplyChainEnv(risk_series)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    model.save("ppo_supply_chain")
    print("Model saved.")

if __name__ == "__main__":
    train_rl_agent()
