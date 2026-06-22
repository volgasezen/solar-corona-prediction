# %%
import os
import zarr

import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sunpy.visualization.colormaps as cm

import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.lib.stride_tricks import sliding_window_view

# %%
# ---------- Load ----------
data = zarr.open("/home/oban/Desktop/Volga/MMI711/sdomlv2.zarr", mode='r')
headers = data.attrs.asdict()
all_image = da.from_zarr(data)

t_obs = np.array(headers["T_OBS"])
df_time = pd.DataFrame(t_obs, columns=["Time"])
df_time["Time"] = pd.to_datetime(df_time["Time"])
idx = df_time.sort_values('Time').index.to_numpy()

# %%
# ---------- Config ----------
INPUT_LEN = 10
PRED_LEN  = 5
WINDOW    = INPUT_LEN + PRED_LEN
GAP_THRESHOLD = 1080      # seconds; anything larger marks a temporal break
N_VAL_PER_ANGLE = 125
ANGLES = list(range(0, 360, 45))

def get_corona_patch(image, angle_deg, radius=206, size=64):
    theta = np.radians(angle_deg)
    cx = int(256 + radius * np.cos(theta))
    cy = int(256 + radius * np.sin(theta))
    half = size // 2
    return image[:, cy - half:cy + half, cx - half:cx + half]

def get_windows_fast(data, window_size, pred_size):
    full_window = window_size + pred_size
    windows = sliding_window_view(data, window_shape=(full_window, 64, 64), axis=(0, 1, 2))
    windows = windows.squeeze()
    return windows[:, :window_size], windows[:, window_size:]

# Time-jump mask once, since all angles share the timestamps
sorted_times = df_time['Time'].iloc[idx].reset_index(drop=True)
diffs_sec = np.r_[0.0, np.diff(sorted_times.values).astype('timedelta64[s]').astype(float)]
breaks = np.where(diffs_sec > 1080)[0]

# %%
train_x_list, train_y_list = [], []
val_x_list,   val_y_list   = [], []

for angle in ANGLES:
    # Exactly your original pipeline, per angle:
    patches = get_corona_patch(all_image, angle)
    sorted_patches = patches[idx]
    x_da, y_da = get_windows_fast(sorted_patches, INPUT_LEN, PRED_LEN)
    x = x_da.compute()                  # NumPy now
    y = y_da.compute()                  # NumPy now

    # Validity mask (corrected indexing — window i covers [i, i+WINDOW))
    n_win = len(x)
    valid = np.ones(n_win, dtype=bool)
    for b in breaks:
        lo = max(0, b - WINDOW + 1)
        hi = min(n_win, b)
        valid[lo:hi] = False
    x, y = x[valid], y[valid]

    # Per-angle train/val split
    train_x_list.append(x[:-N_VAL_PER_ANGLE])
    train_y_list.append(y[:-N_VAL_PER_ANGLE])
    val_x_list.append(x[-N_VAL_PER_ANGLE:])
    val_y_list.append(y[-N_VAL_PER_ANGLE:])
    print(f"angle {angle}: {len(x)} valid windows")

# NumPy concat — no dask involved
train_x = np.concatenate(train_x_list, axis=0)
train_y = np.concatenate(train_y_list, axis=0)
val_x   = np.concatenate(val_x_list,   axis=0)
val_y   = np.concatenate(val_y_list,   axis=0)

print("train:", train_x.shape, train_y.shape)
print("val:",   val_x.shape,   val_y.shape)
# %%


# %%
log_train_x = np.log1p(train_x)
log_min = np.percentile(log_train_x, 0.5)
log_max = np.percentile(log_train_x, 99.5)
scale = log_max - log_min

def normalize(arr):
    return np.clip((np.log1p(arr) - log_min) / scale, 0.0, 1.0).astype(np.float32)

train_x = normalize(train_x)
train_y = normalize(train_y)
val_x   = normalize(val_x)
val_y   = normalize(val_y)
