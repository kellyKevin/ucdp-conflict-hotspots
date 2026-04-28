import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import wbgapi as wb
from sklearn.cluster import DBSCAN
from stable_baselines3 import PPO
from rl_env import ConflictSupplyChainEnv

st.set_page_config(page_title="Conflict Pattern Analysis", layout="wide")
st.title("🌍 Conflict Pattern Analysis & Risk Forecasting")

@st.cache_data
def get_data():
    return pd.read_pickle('cleaned_data.pkl')

df = get_data()
selected_region = st.sidebar.selectbox("Select Region", df['region'].unique())
filtered_df = df[df['region'] == selected_region]

st.header("1. EDA")
col1, col2 = st.columns(2)
with col1:
    yearly_counts = filtered_df.groupby('year')['id'].count().reset_index()
    st.plotly_chart(px.line(yearly_counts, x='year', y='id', title="Event Count"), width='stretch')
with col2:
    yearly_fatalities = filtered_df.groupby('year')['best'].sum().reset_index()
    st.plotly_chart(px.line(yearly_fatalities, x='year', y='best', title="Fatalities"), width='stretch')

st.header("2. Clustering")
if st.button("Run Clustering"):
    recent_df = filtered_df[filtered_df['year'] >= 2020].copy()
    db = DBSCAN(eps=0.5, min_samples=10).fit(recent_df[['latitude', 'longitude']])
    recent_df['cluster'] = db.labels_
    st.plotly_chart(px.scatter_mapbox(recent_df[recent_df['cluster'] != -1], lat="latitude", lon="longitude", color="cluster", mapbox_style="carto-positron"), width='stretch')

st.header("3. Forecast")
selected_country = st.selectbox("Select Country", filtered_df['country'].unique())
if st.button("Forecast"):
    c_df = df[df['country'] == selected_country].copy()
    c_df['ds'] = c_df['date_end'].dt.to_period('M').dt.to_timestamp()
    ts = c_df.groupby('ds')['best'].sum().reset_index()
    ts.columns = ['ds', 'y']
    m = Prophet().fit(ts)
    future = m.make_future_dataframe(periods=12, freq='ME')
    forecast = m.predict(future)
    st.plotly_chart(px.line(forecast, x='ds', y='yhat', title="Fatalities Forecast"), width='stretch')

st.header("4. RL Simulation")
if st.button("Simulate Decision Making"):
    try:
        model = PPO.load("ppo_supply_chain")
        c_df = df[df['country'] == selected_country].tail(24).copy()
        ts = c_df.groupby(c_df['date_end'].dt.to_period('M'))['best'].sum()
        risk_series = ts / (df[df['country'] == selected_country].groupby(df['date_end'].dt.to_period('M'))['best'].sum().max() + 1e-6)
        env = ConflictSupplyChainEnv(risk_series)
        obs, _ = env.reset()
        res = []
        for i in range(len(risk_series)):
            action, _ = model.predict(obs, deterministic=True)
            res.append({"Month": str(ts.index[i]), "Action": ["Route Normally", "Reroute", "Halt"][action]})
            obs, _, _, _, _ = env.step(action)
        st.table(pd.DataFrame(res))
    except:
        st.error("Model not found. Run train_rl.py first.")
