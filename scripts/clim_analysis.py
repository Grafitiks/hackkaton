import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
print("Non-null climatology in train:", df['ndvi_climatology_mean'].notna().sum())
print("Null climatology in train:", df['ndvi_climatology_mean'].isna().sum())

# In train, when is ndvi_climatology_mean non-null?
print("When primary_ndvi is notna, climatology is notna:", df[df['primary_ndvi'].notna()]['ndvi_climatology_mean'].notna().sum())
print("When primary_ndvi is isna, climatology is notna:", df[df['primary_ndvi'].isna()]['ndvi_climatology_mean'].notna().sum())

# How is ndvi_climatology_mean related to doy and anon_polygon_id?
# Is it smooth? Does it vary by year or only by doy and polygon?
p2 = df[df['anon_polygon_id'] == 'AOI-0002']
p2_notna = p2[p2['ndvi_climatology_mean'].notna()]
print("\nAOI-0002 unique years:", p2_notna['year'].unique())
doy_check = p2_notna.groupby('doy')['ndvi_climatology_mean'].agg(['min', 'max', 'std', 'count'])
print("For AOI-0002, does climatology_mean vary across different years for the same DOY?")
print("Max std across years for same DOY:", doy_check['std'].max())
print(doy_check.head(10))
