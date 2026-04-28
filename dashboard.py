import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import wbgapi as wb
from sklearn.cluster import DBSCAN
from stable_baselines3 import PPO
from rl_env import ConflictSupplyChainEnv
import os

st.set_page_config(page_title="Conflict Pattern Analysis Dashboard", layout="wide")

st.title("🌍 Global Conflict Analysis & Corporate Risk AI")
st.markdown("""
### Data-Driven Insights into Global Organized Violence
This dashboard leverages the **UCDP Georeferenced Event Dataset (GED)** and **World Bank** indicators to analyze
conflict hotspots, forecast fatality trends, and simulate strategic business decisions using Reinforcement Learning.
""")

# --- Sidebar ---
st.sidebar.image("https://ucdp.uu.se/assets/images/logo.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Research Insights"])

@st.cache_data
def get_data():
    if os.path.exists('cleaned_data.pkl'):
        return pd.read_pickle('cleaned_data.pkl')
    return None

df = get_data()

if df is None:
    st.error("Data not found. Please run `data_prep.py` first.")
    st.stop()

if page == "Dashboard":
    selected_region = st.sidebar.selectbox("Select Region", df['region'].unique())
    filtered_df = df[df['region'] == selected_region]

    tabs = st.tabs(["📊 Exploratory Analysis", "📍 Hot Zone Detection", "📈 Fatality Forecast", "💰 Economic Correlation", "🤖 AI Simulation"])

    with tabs[0]:
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            yearly_counts = filtered_df.groupby('year')['id'].count().reset_index()
            fig_freq = px.line(yearly_counts, x='year', y='id', title="Conflict Frequency Over Time",
                               labels={'id': 'Event Count', 'year': 'Year'}, template="plotly_white")
            st.plotly_chart(fig_freq, width='stretch')
        with col2:
            yearly_fatalities = filtered_df.groupby('year')['best'].sum().reset_index()
            fig_fat = px.line(yearly_fatalities, x='year', y='best', title="Fatalities Over Time",
                              labels={'best': 'Total Fatalities', 'year': 'Year'}, template="plotly_white")
            st.plotly_chart(fig_fat, width='stretch')

        st.subheader("Geographic Event Map")
        fig_map = px.scatter_mapbox(filtered_df.sample(min(len(filtered_df), 3000)),
                                    lat="latitude", lon="longitude", color="best",
                                    hover_name="country", size="best",
                                    color_continuous_scale=px.colors.sequential.Reds, size_max=15, zoom=1,
                                    mapbox_style="carto-positron", height=600)
        st.plotly_chart(fig_map, width='stretch')

    with tabs[1]:
        st.header("Conflict Hot Zones (DBSCAN)")
        st.write("DBSCAN clustering identifies high-density areas of recent violence (2020+).")
        if st.button("Detect Hot Zones"):
            recent_df = filtered_df[filtered_df['year'] >= 2020].copy()
            if not recent_df.empty:
                db = DBSCAN(eps=0.5, min_samples=10).fit(recent_df[['latitude', 'longitude']])
                recent_df['cluster'] = db.labels_
                fig_clusters = px.scatter_mapbox(recent_df[recent_df['cluster'] != -1],
                                                lat="latitude", lon="longitude", color="cluster",
                                                hover_name="country", zoom=1, mapbox_style="carto-positron",
                                                title="High-Intensity Clusters identified", height=600)
                st.plotly_chart(fig_clusters, width='stretch')
            else:
                st.warning("No recent data available for this region.")

    with tabs[2]:
        st.header("Fatality Trend Forecasting")
        selected_country = st.selectbox("Select Country", filtered_df['country'].unique())
        if st.button("Generate Prophet Forecast"):
            with st.spinner("Calculating forecast..."):
                c_df = df[df['country'] == selected_country].copy()
                c_df['ds'] = c_df['date_end'].dt.to_period('M').dt.to_timestamp()
                ts = c_df.groupby('ds')['best'].sum().reset_index()
                ts.columns = ['ds', 'y']
                m = Prophet().fit(ts)
                future = m.make_future_dataframe(periods=12, freq='ME')
                forecast = m.predict(future)
                fig_f = px.line(forecast, x='ds', y='yhat', title=f"12-Month Fatality Forecast for {selected_country}",
                                labels={'yhat': 'Predicted Fatalities', 'ds': 'Date'})
                st.plotly_chart(fig_f, width='stretch')

    with tabs[3]:
        st.header("Economics & Conflict")
        iso_map = {'Afghanistan': 'AFG', 'Ukraine': 'UKR', 'Mexico': 'MEX', 'Syria': 'SYR', 'Iraq': 'IRQ', 'Sudan': 'SDN', 'Nigeria': 'NGA'}
        if selected_country in iso_map:
            iso = iso_map[selected_country]
            st.subheader(f"GDP Growth vs Conflict Intensity: {selected_country}")
            econ_df = wb.data.DataFrame('NY.GDP.MKTP.KD.ZG', iso, time=range(1989, 2024)).T
            econ_df.index = econ_df.index.str.replace('YR', '').astype(int)
            conf_y = df[df['country'] == selected_country].groupby('year')['best'].sum()
            merged = pd.concat([econ_df, conf_y], axis=1).fillna(0)
            merged.columns = ['GDP_Growth', 'Fatalities']
            fig_e = px.scatter(merged, x='GDP_Growth', y='Fatalities', trendline="ols",
                               title="Relationship between GDP Growth and Fatalities")
            st.plotly_chart(fig_e, width='stretch')
        else:
            st.info("Select a country with available economic mapping (e.g., Afghanistan, Ukraine, Mexico).")

    with tabs[4]:
        st.header("AI Supply Chain Simulation")
        st.write("A Reinforcement Learning (PPO) agent makes real-time decisions based on conflict risk.")
        if st.button("Execute Simulation"):
            if os.path.exists("ppo_supply_chain.zip"):
                model = PPO.load("ppo_supply_chain")
                c_df = df[df['country'] == selected_country].tail(24).copy()
                ts = c_df.groupby(c_df['date_end'].dt.to_period('M'))['best'].sum()
                max_val = df[df['country'] == selected_country].groupby(df['date_end'].dt.to_period('M'))['best'].sum().max()
                risk_series = ts / (max_val + 1e-6)
                env = ConflictSupplyChainEnv(risk_series)
                obs, _ = env.reset()
                res = []
                for i in range(len(risk_series)):
                    action, _ = model.predict(obs, deterministic=True)
                    action_name = ["Route Normally", "Reroute", "Halt"][action]
                    res.append({"Month": str(ts.index[i]), "Risk Score": round(float(risk_series.iloc[i]), 3), "AI Action": action_name})
                    obs, _, _, _, _ = env.step(action)
                st.table(pd.DataFrame(res))
                st.success("Simulation complete. The agent learns to reroute or halt as risk scores escalate.")
            else:
                st.error("RL Model not found. Please run `train_rl.py` first.")

elif page == "Research Insights":
    st.header("Academic & Policy Research")
    if os.path.exists("insights.md"):
        with open("insights.md", "r") as f:
            st.markdown(f.read())
    else:
        st.write("Research insights file missing.")
