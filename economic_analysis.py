import pandas as pd
import wbgapi as wb
import matplotlib.pyplot as plt

def economic_correlation(country_code, country_name):
    indicators = {'NY.GDP.MKTP.KD.ZG': 'GDP_Growth', 'FP.CPI.TOTL.ZG': 'Inflation'}
    econ_df = wb.data.DataFrame(indicators.keys(), country_code, time=range(1989, 2024)).T
    econ_df.columns = indicators.values()
    econ_df.index = econ_df.index.str.replace('YR', '').astype(int)

    df = pd.read_pickle('cleaned_data.pkl')
    conflict_yearly = df[df['country'] == country_name].groupby('year')['best'].sum()
    merged = pd.concat([econ_df, conflict_yearly], axis=1).fillna(0)
    merged.columns = ['GDP_Growth', 'Inflation', 'fatalities']

    print(merged.corr())

    plt.figure(figsize=(10, 6))
    plt.scatter(merged['GDP_Growth'], merged['fatalities'])
    plt.title(f'GDP Growth vs Fatalities: {country_name}')
    plt.savefig(f'econ_corr_{country_name}.png')

if __name__ == "__main__":
    economic_correlation('AFG', 'Afghanistan')
