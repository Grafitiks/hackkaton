import sys
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator

sys.stdout.reconfigure(encoding='utf-8')

print("=== BENCHMARKING RECONSTRUCTION METHODS ===")

# Load train dataset
df_train = pd.read_csv("data/train_dataset.csv", encoding='utf-8')
df_train['date_dt'] = pd.to_datetime(df_train['date'])
df_train = df_train.sort_values(['anon_polygon_id', 'date_dt']).reset_index(drop=True)

# Select all known primary_ndvi indices
known_idx = df_train[df_train['primary_ndvi'].notna()].index.values
print(f"Total known primary_ndvi points in train: {len(known_idx)}")

# Set random seed
np.random.seed(42)

# Create a validation mask mimicking test:
# In test, ~3112 / 20753 ~ 15% of all observation points were selected as synthetic gaps.
# Mostly length 1, occasionally length 2.
# Let's sample ~15% of known indices as validation synthetic gaps.
val_mask = np.zeros(len(df_train), dtype=bool)

# Sample 4500 points from known_idx
sampled_val_idx = np.random.choice(known_idx, size=4500, replace=False)
val_mask[sampled_val_idx] = True

y_true = df_train.loc[val_mask, 'primary_ndvi'].values
print(f"Validation sample size: {len(y_true)}")

def calc_metrics(y_t, y_p, name="Method"):
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    gap_score = round(float(30.0 * max(0.0, 1.0 - rmse / 0.10)), 2)
    mae = np.mean(np.abs(y_t - y_p))
    print(f"{name:30s} | RMSE: {rmse:.5f} | GapScore: {gap_score:5.2f} / 30 | MAE: {mae:.5f}")
    return rmse, gap_score

# 1. Linear interpolation on the time series of each polygon
# For validation, we mask out the sampled_val_idx
df_sim = df_train[['anon_polygon_id', 'date_dt', 'primary_ndvi']].copy()
df_sim.loc[val_mask, 'primary_ndvi'] = np.nan

# Linear interpolation by polygon
df_sim['ndvi_linear'] = df_sim.groupby('anon_polygon_id')['primary_ndvi'].transform(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)

y_pred_linear = df_sim.loc[val_mask, 'ndvi_linear'].values
calc_metrics(y_true, y_pred_linear, "Linear Interpolation (all-time)")

# Linear interpolation WITHIN each polygon & YEAR (season)
# Since winter is not in the data, interpolating across October 30 -> April 01 could be dangerous
df_sim['year'] = df_sim['date_dt'].dt.year
df_sim['ndvi_linear_seasonal'] = df_sim.groupby(['anon_polygon_id', 'year'])['primary_ndvi'].transform(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
# If edges within year are nan, fallback to all-time linear
mask_nan = df_sim['ndvi_linear_seasonal'].isna()
df_sim.loc[mask_nan, 'ndvi_linear_seasonal'] = df_sim.loc[mask_nan, 'ndvi_linear']

y_pred_linear_seas = df_sim.loc[val_mask, 'ndvi_linear_seasonal'].values
calc_metrics(y_true, y_pred_linear_seas, "Linear Interpolation (seasonal)")

# Time-weighted interpolation (exact days)
# Since the grid is daily, time-weighted is identical to linear on daily index.
# Let's test PCHIP / Akima / Spline
pchip_preds = []
akima_preds = []

for (poly, yr), group in df_sim.groupby(['anon_polygon_id', 'year']):
    val_in_group = val_mask[group.index]
    if not val_in_group.any():
        continue
    
    # known in group
    known_in_group = group[group['primary_ndvi'].notna()]
    if len(known_in_group) >= 4:
        x_kn = (known_in_group['date_dt'] - pd.to_datetime(f"{yr}-01-01")).dt.days.values
        y_kn = known_in_group['primary_ndvi'].values
        
        # PCHIP
        pchip = PchipInterpolator(x_kn, y_kn, extrapolate=False)
        x_all = (group['date_dt'] - pd.to_datetime(f"{yr}-01-01")).dt.days.values
        df_sim.loc[group.index, 'ndvi_pchip'] = pchip(x_all)
        
        # Akima
        try:
            akima = Akima1DInterpolator(x_kn, y_kn)
            df_sim.loc[group.index, 'ndvi_akima'] = akima(x_all)
        except Exception:
            df_sim.loc[group.index, 'ndvi_akima'] = np.nan
    else:
        df_sim.loc[group.index, 'ndvi_pchip'] = np.nan
        df_sim.loc[group.index, 'ndvi_akima'] = np.nan

# Fallback for pchip and akima to linear
df_sim['ndvi_pchip'] = df_sim['ndvi_pchip'].fillna(df_sim['ndvi_linear_seasonal'])
df_sim['ndvi_akima'] = df_sim['ndvi_akima'].fillna(df_sim['ndvi_linear_seasonal'])

y_pred_pchip = df_sim.loc[val_mask, 'ndvi_pchip'].values
calc_metrics(y_true, y_pred_pchip, "PCHIP Monotonic Spline")

y_pred_akima = df_sim.loc[val_mask, 'ndvi_akima'].values
calc_metrics(y_true, y_pred_akima, "Akima Spline")
