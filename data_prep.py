import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    relevant_columns = [
        'id', 'year', 'type_of_violence', 'side_a', 'side_b',
        'latitude', 'longitude', 'country', 'region', 'date_start', 'date_end', 'best'
    ]
    df = df[relevant_columns]

    df['best'] = df['best'].fillna(0)
    df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
    df['date_end'] = pd.to_datetime(df['date_end'], errors='coerce')
    df = df.dropna(subset=['date_start', 'date_end'])

    print("Data cleaning complete.")
    return df

if __name__ == "__main__":
    df = load_and_clean_data('GEDEvent_v25_1.csv')
    df.to_pickle('cleaned_data.pkl')
    print("Cleaned data saved to cleaned_data.pkl")
