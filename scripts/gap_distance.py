import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

df_test = pd.read_csv("data/private_features.csv", encoding='utf-8')
df_test['date_dt'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date_dt'].dt.year

# Let's inspect for each polygon how primary_ndvi is distributed
distances_prev = []
distances_next = []

for poly_id, group in df_test.groupby('anon_polygon_id'):
    group = group.sort_values('date_dt').copy()
    
    # We want to know: among the dates where primary_ndvi is known (not NaN and is_synthetic_gap == False)
    known_idx = group[group['primary_ndvi'].notna()].index
    known_dates = group.loc[known_idx, 'date_dt'].values
    
    gap_rows = group[group['is_synthetic_gap'] == True]
    for _, row in gap_rows.iterrows():
        g_date = row['date_dt']
        g_year = row['year']
        
        # Previous known date in same year or overall
        prev_dates = known_dates[known_dates < np.datetime64(g_date)]
        next_dates = known_dates[known_dates > np.datetime64(g_date)]
        
        d_prev = (g_date - pd.to_datetime(prev_dates[-1])).days if len(prev_dates) > 0 else 999
        d_next = (pd.to_datetime(next_dates[0]) - g_date).days if len(next_dates) > 0 else 999
        
        distances_prev.append(d_prev)
        distances_next.append(d_next)

d_prev_s = pd.Series(distances_prev)
d_next_s = pd.Series(distances_next)
min_dist_s = pd.Series(np.minimum(distances_prev, distances_next))

print("Distance in days to previous known primary_ndvi:")
print(d_prev_s.describe())
print("\nDistance in days to next known primary_ndvi:")
print(d_next_s.describe())
print("\nMinimum distance to either neighbor (days):")
print(min_dist_s.describe())
print("\nMin distance quantiles:")
print(min_dist_s.value_counts().head(15).sort_index())
