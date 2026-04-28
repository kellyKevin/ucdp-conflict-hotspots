import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    df = pd.read_pickle('cleaned_data.pkl')
    yearly_stats = df.groupby('year').agg({'id': 'count', 'best': 'sum'}).rename(columns={'id': 'event_count', 'best': 'fatalities'})

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=yearly_stats, x='year', y='event_count')
    plt.title('Conflict Frequency Over Time')

    plt.subplot(1, 2, 2)
    sns.lineplot(data=yearly_stats, x='year', y='fatalities')
    plt.title('Fatalities Over Time')
    plt.tight_layout()
    plt.savefig('temporal_trends.png')

    top_countries = df.groupby('country')['best'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 countries by fatalities:")
    print(top_countries)

if __name__ == "__main__":
    perform_eda()
