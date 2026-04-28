import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def detect_hot_zones():
    df = pd.read_pickle('cleaned_data.pkl')
    recent_df = df[df['year'] >= 2020].copy()
    coords = recent_df[['latitude', 'longitude']].values
    db = DBSCAN(eps=0.5, min_samples=10).fit(coords)
    recent_df['cluster'] = db.labels_

    plt.figure(figsize=(12, 8))
    plt.scatter(recent_df['longitude'], recent_df['latitude'], c=recent_df['cluster'], cmap='tab20', s=1, alpha=0.5)
    plt.title('Conflict Clusters (2020-Present)')
    plt.savefig('clusters.png')

if __name__ == "__main__":
    detect_hot_zones()
