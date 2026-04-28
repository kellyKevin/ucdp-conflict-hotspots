import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast_region(country_name):
    df = pd.read_pickle('cleaned_data.pkl')
    country_df = df[df['country'] == country_name].copy()
    country_df['ds'] = country_df['date_end'].dt.to_period('M').dt.to_timestamp()
    ts = country_df.groupby('ds')['best'].sum().reset_index()
    ts.columns = ['ds', 'y']

    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=24, freq='ME')
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f'Fatality Forecast for {country_name}')
    plt.savefig(f'forecast_{country_name}.png')
    return forecast

if __name__ == "__main__":
    forecast_region('Afghanistan')
