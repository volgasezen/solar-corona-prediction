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
INPUT_LEN = 15
WINDOW    = INPUT_LEN
GAP_THRESHOLD = 1080      # seconds; anything larger marks a temporal break
N_VAL_PER_ANGLE = 125
ANGLES = list(range(0, 360, 45))

def get_corona_patch(image, angle_deg, radius=206, size=64):
    theta = np.radians(angle_deg)
    cx = int(256 + radius * np.cos(theta))
    cy = int(256 + radius * np.sin(theta))
    half = size // 2
    return image[:, cy - half:cy + half, cx - half:cx + half]

# ---------- Build per-angle frame arrays (no windowing yet) ----------
# Each per-angle array is (N_time, 64, 64). For 8 angles, total: 8 * ~6000 * 64 * 64 * 4 = ~750 MB.

frame_arrays = []           # list of (N_time_a, 64, 64) NumPy arrays, one per angle
angle_labels = []           # parallel list, integer label per angle

for angle_id, angle in enumerate(ANGLES):
    patches = get_corona_patch(all_image, angle)
    sorted_patches = patches[idx].compute()      # (N_time, 64, 64) NumPy
    frame_arrays.append(sorted_patches.astype(np.float32))
    angle_labels.append(angle_id)
print('1')
# %%
# ---------- Normalize in-place to save memory ----------
# Compute stats from training portion only. We need to know the split BEFORE the dataset is built.

log_min, log_max = np.load('/home/oban/Desktop/Volga/MMI711/project/train_min_max.npy')
scale = log_max - log_min

# Apply log + min-max in place, per angle, to avoid duplicate arrays
for i in range(len(frame_arrays)):
    np.log1p(frame_arrays[i], out=frame_arrays[i])   # in-place
    frame_arrays[i] -= log_min
    frame_arrays[i] /= scale
    np.clip(frame_arrays[i], 0.0, 1.0, out=frame_arrays[i])
print('2')
# %%
# ---------- Build list of (angle_idx, start_idx, is_val) for valid windows ----------

# Time-jump mask (same for all angles since timestamps are shared)
sorted_times = df_time['Time'].iloc[idx].reset_index(drop=True)
diffs_sec = np.r_[0.0, np.diff(sorted_times.values).astype('timedelta64[s]').astype(float)]
breaks = np.where(diffs_sec > 1080)[0]

train_index, val_index = [], []
for angle_id, fa in enumerate(frame_arrays):
    n_time = len(fa)
    n_win = n_time - WINDOW + 1

    # Validity mask for window starting positions
    valid = np.ones(n_win, dtype=bool)
    for b in breaks:
        lo = max(0, b - WINDOW + 1)
        hi = min(n_win, b)
        valid[lo:hi] = False

    valid_starts = np.where(valid)[0]
    # Per-angle temporal split: last N_VAL_PER_ANGLE go to val
    train_starts = valid_starts[:-N_VAL_PER_ANGLE][::WINDOW]
    val_starts   = valid_starts[-N_VAL_PER_ANGLE:][::WINDOW]

    train_index.extend([(angle_id, s) for s in train_starts])
    val_index.extend  ([(angle_id, s) for s in val_starts])
print('3')
# ---------- Dataset ----------
class SDOSeqDataset(torch.utils.data.Dataset):
    def __init__(self, frame_arrays, index_list, input_len):
        # Convert each angle's NumPy frames to a torch tensor (still on CPU).
        # torch.from_numpy shares memory with NumPy — no copy.
        self.frames = [torch.from_numpy(fa) for fa in frame_arrays]
        self.index = index_list           # list of (angle_id, start_idx)
        self.in_len = input_len

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        angle_id, s = self.index[i]
        clip = self.frames[angle_id][s : s + self.in_len]   # (W, 64, 64) — view
        clip = clip.unsqueeze(1)
        return {
            "x": clip,
            "angle_id": torch.tensor(angle_id, dtype=torch.long),
        }

train_ds = SDOSeqDataset(frame_arrays, train_index, INPUT_LEN)
val_ds   = SDOSeqDataset(frame_arrays, val_index,   INPUT_LEN)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)
print('Done')

# %%

# %%
class CNN_VAE(nn.Module):
    def __init__(self, base=32, latent_ch=128, bottleneck_ch=16):
        super().__init__()

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
            )

        self.encoder = nn.Sequential(
            down(1, base),          # 64 -> 32
            down(base, base * 2),   # 32 -> 16
            down(base * 2, latent_ch),  # 16 -> 8
            nn.Conv2d(latent_ch, bottleneck_ch, 1),
            nn.GroupNorm(1, bottleneck_ch),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_ch, latent_ch, 1),
            up(latent_ch, base * 2),    # 8 -> 16
            up(base * 2, base),         # 16 -> 32
            up(base, base),             # 32 -> 64
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid(),  # assumes inputs in [0,1] after log+minmax
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z
# %%

model = CNN_VAE().cuda()

#x = torch.randn((128,15,1,64,64)).cuda()
#x = x.view(128 * 15, 1, 64, 64)
#model(x)[0].shape

# %%
from pytorch_msssim import ssim
import torch.nn.functional as F

def reconstruction_loss(pred, target):
    l1   = F.l1_loss(pred, target)
    ssi = 1 - ssim(pred, target, data_range=1.0, size_average=True, win_size=11)
    return 0.2 * l1 + 0.8 * ssi

def spatio_temporal_loss(mu_seq, T, step_size=1):
    """
    Optimized, loop-free temporal distance loss.
    mu_seq: Tensor of shape (Batch, Seq_Len, Channels, Height, Width)
    """
    B_times_T, C, H, W = mu_seq.shape
    B = B_times_T//T
    mu_seq = mu_seq.view(B, T, -1)
    # 1. Compute all pairwise L2 distances in the sequence simultaneously
    # dists shape: (Batch, Seq_Len, Seq_Len)
    dists = torch.cdist(mu_seq, mu_seq, p=2)

    # 2. Create the target distance matrix
    # time_steps: [0, 1, 2, ..., T-1]
    time_steps = torch.arange(T, device=mu_seq.device, dtype=torch.float32)

    # Broadcast subtraction to get absolute time differences between all pairs
    # target_matrix shape: (T, T)
    time_diffs = torch.abs(time_steps.unsqueeze(0) - time_steps.unsqueeze(1))
    target_matrix = time_diffs * step_size

    # Expand target matrix to match batch size: (Batch, T, T)
    target_matrix = target_matrix.unsqueeze(0).expand(B, -1, -1)

    # 3. We only care about the upper triangle (forward in time, where t < t+k)
    # This boolean mask extracts only the relevant pairs, ignoring self-distance (diagonal)
    mask = torch.triu(torch.ones(T, T, device=mu_seq.device), diagonal=1).bool()

    # 4. Extract valid actual distances and target distances
    # Shapes become (Batch, number_of_valid_pairs)
    actual_dists = dists[:, mask]
    target_dists = target_matrix[:, mask]

    return F.huber_loss(actual_dists, target_dists, delta=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

import math

def get_annealed_weight(epoch, start_epoch, cycle_length, max_weight, ramp_ratio=0.5):
    """
    Linearly scales a weight from 0 to max_weight between start_epoch and end_epoch.
    """
    if epoch < start_epoch:
        return 0.0
    step_in_cycle = (epoch - start_epoch) % cycle_length
    ramp_epochs = int(cycle_length * ramp_ratio)

    if step_in_cycle < ramp_epochs:
        # Linear ramp phase
        progress = step_in_cycle / ramp_epochs
        return max_weight * progress
    else:
        # Plateau phase (hold steady at max_weight)
        return max_weight

def train(model, current_beta, current_lambda):
    model.train()
    l = []
    for x in train_loader:
        x = x['x'].cuda()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        preds, z = model(x)
        loss_recon = reconstruction_loss(preds, x)
        loss_temp = spatio_temporal_loss(z, T)
        #total_loss = loss_recon + (current_lambda * loss_temp)
        # Backward
        optimizer.zero_grad()
        loss_recon.backward()
        optimizer.step()
        l.append([loss_recon.item()])
    return l

def valid(model):
    l = []
    model.eval()
    with torch.no_grad():
        for x in val_loader:
            x = x['x'].cuda()
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            preds, z = model(x)
            loss_recon = reconstruction_loss(preds, x)
            loss_temp = spatio_temporal_loss(z, T)
            l.append([loss_recon.item(),loss_temp.item()])
    return l

# %%

#a = [[2,3,4],[0,100,2333]]
#import numpy as np
#print(f'Loss values: {np.mean(a, axis=0)}')

# %%
max_beta = 0.005
max_lambda_temp = 0.1

for i in range(200):
    current_beta = get_annealed_weight(i, start_epoch=20, cycle_length=30, max_weight=max_beta)
    current_lambda = get_annealed_weight(i, start_epoch=20, cycle_length=30, max_weight=max_lambda_temp)
    train_l = train(model, current_beta, current_lambda)
    val_l = valid(model)
    if i%1 == 0:
        print(f'Epoch {i} done.')
        print(f'Train loss: {np.mean(train_l, axis=0)}')
        print(f'Val loss: {np.mean(val_l, axis=0)}')

torch.save(model.state_dict(), '/home/oban/Desktop/Volga/MMI711/project/3-lstm2/model_conv_ae3.pt')
