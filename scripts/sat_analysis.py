import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
known = df[df['primary_ndvi'].notna()].copy()
print(f"Total rows with primary_ndvi: {len(known)}")

# When s2_ndvi is present
s2 = known[known['s2_ndvi'].notna()]
print(f"Rows with s2_ndvi: {len(s2)}")
print(f"Diff |primary_ndvi - s2_ndvi| mean: {(s2['primary_ndvi'] - s2['s2_ndvi']).abs().mean():.6f}, max: {(s2['primary_ndvi'] - s2['s2_ndvi']).abs().max():.6f}")

# When landsat_ndvi is present
ls = known[known['landsat_ndvi'].notna()]
print(f"Rows with landsat_ndvi: {len(ls)}")
print(f"Diff |primary_ndvi - landsat_ndvi| mean: {(ls['primary_ndvi'] - ls['landsat_ndvi']).abs().mean():.6f}, max: {(ls['primary_ndvi'] - ls['landsat_ndvi']).abs().max():.6f}")

# When modis_ndvi is present
mod = known[known['modis_ndvi'].notna()]
print(f"Rows with modis_ndvi: {len(mod)}")
print(f"Diff |primary_ndvi - modis_ndvi| mean: {(mod['primary_ndvi'] - mod['modis_ndvi']).abs().mean():.6f}, max: {(mod['primary_ndvi'] - mod['modis_ndvi']).abs().max():.6f}")

# Check which satellite is primary when multiple are present or what hierarchy
both_s2_ls = known[known['s2_ndvi'].notna() & known['landsat_ndvi'].notna()]
print(f"Rows with both s2 and landsat: {len(both_s2_ls)}")
if len(both_s2_ls) > 0:
    diff_s2 = (both_s2_ls['primary_ndvi'] - both_s2_ls['s2_ndvi']).abs()
    diff_ls = (both_s2_ls['primary_ndvi'] - both_s2_ls['landsat_ndvi']).abs()
    print("Match s2 exactly (<1e-5):", (diff_s2 < 1e-5).sum(), "Match ls exactly (<1e-5):", (diff_ls < 1e-5).sum())

# How is primary_ndvi generated when s2, landsat, modis are all NaN?
no_sat = known[known['s2_ndvi'].isna() & known['landsat_ndvi'].isna() & known['modis_ndvi'].isna()]
print(f"Rows with primary_ndvi but ALL sat indices are NaN: {len(no_sat)}")

# What satellites are present when primary_ndvi is present?
print("Satellite presence counts:")
print("s2 only:", (known['s2_ndvi'].notna() & known['landsat_ndvi'].isna() & known['modis_ndvi'].isna()).sum())
print("landsat only:", (known['s2_ndvi'].isna() & known['landsat_ndvi'].notna() & known['modis_ndvi'].isna()).sum())
print("modis only:", (known['s2_ndvi'].isna() & known['landsat_ndvi'].isna() & known['modis_ndvi'].notna()).sum())
print("any sat present:", (known['s2_ndvi'].notna() | known['landsat_ndvi'].notna() | known['modis_ndvi'].notna()).sum())
