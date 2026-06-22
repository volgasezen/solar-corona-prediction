# %%
import os
import zarr;

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
def get_corona_patch(image, angle_deg, radius=206, size=64):
    # Convert angle to radians
    theta = np.radians(angle_deg)

    # Calculate patch center
    cx = int(256 + radius * np.cos(theta))
    cy = int(256 + radius * np.sin(theta))

    # Extract patch (with boundary padding if necessary)
    half = size // 2
    patch = image[:, cy-half:cy+half, cx-half:cx+half]

    return patch

# %%
t_obs = np.array(headers["T_OBS"])
df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
df_time["Time"] = pd.to_datetime(df_time["Time"])

idx = df_time.sort_values('Time').index

patches = da.concatenate(
    [get_corona_patch(all_image, a)[idx] for a in range(0, 360, 45)], axis=0
).compute()

patches.shape
# %%
log_min, log_max = np.load('/home/oban/Desktop/Volga/MMI711/project/train_min_max.npy')

np.log1p(patches, out=patches)

patches = np.clip((patches-log_min)/(log_max-log_min), 0, 1)

# %%
np.max(patches)

# %%

class CNN_AE(nn.Module):
    def __init__(self, base=32, latent_ch=128):
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
        )
        self.decoder = nn.Sequential(
            up(latent_ch, base * 2),    # 8 -> 16
            up(base * 2, base),         # 16 -> 32
            up(base, base),             # 32 -> 64
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid(),  # assumes inputs in [0,1] after log+minmax
        )

    def encode(self, x):
        # x: (B, 1, 64, 64) -- handle channel dim in the dataset, not here
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

model = CNN_AE().cuda()

state_dict = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_conv_ae2.pt', weights_only=False)
model.load_state_dict(state_dict)
# %%
len(patches)

# %%

import torch
import numpy as np

# Assumes:
#   frame_arrays: list of (N_time_a, 64, 64) float32 NumPy arrays, normalized to [0, 1]
#   model: your trained CNN_AE, loaded on cuda, in eval mode

model = model.cuda().eval()

@torch.no_grad()
def encode_frames(frames_np, batch_size=512):
    """frames_np: (N, 64, 64) float32 in [0, 1]. Returns (N, C, h, w) float32 NumPy."""
    n = len(frames_np)
    print(n)
    out_chunks = []
    for i in range(0, n, batch_size):
        batch = frames_np[i:i + batch_size]
        #print(batch.shape)
        # Add channel dim and move to GPU
        x = torch.from_numpy(batch).unsqueeze(1).cuda()
        #print(x.shape) # (B, 1, 64, 64)
        z = model.encode(x)                              # (B, C, h, w)
        out_chunks.append(z.cpu().numpy().astype(np.float32))
    return np.vstack(out_chunks)

# Replace frame_arrays in place with latent_arrays
latent_arrays = encode_frames(patches).reshape(8, 6135, 128, 8, 8)

print(f"Latent shape per frame: {latent_arrays[0].shape[1:]}")
print(f"Total latents: {sum(len(z) for z in latent_arrays)}")

# Optional: free the frame data if you don't need it anymore
#del frame_arrays
print(np.array(latent_arrays).shape)
# %%
INPUT_LEN = 10
PRED_LEN  = 5
WINDOW    = INPUT_LEN + PRED_LEN
GAP_THRESHOLD = 1080      # seconds; anything larger marks a temporal break
N_VAL_PER_ANGLE = 125
ANGLES = list(range(0, 360, 45))

sorted_times = df_time['Time'].iloc[idx].reset_index(drop=True)
diffs_sec = np.r_[0.0, np.diff(sorted_times.values).astype('timedelta64[s]').astype(float)]
breaks = np.where(diffs_sec > 1080)[0]

train_index, val_index = [], []
for angle_id, fa in enumerate(latent_arrays):
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
    train_starts = valid_starts[:-N_VAL_PER_ANGLE]
    val_starts   = valid_starts[-N_VAL_PER_ANGLE:]

    train_index.extend([(angle_id, s) for s in train_starts])
    val_index.extend  ([(angle_id, s) for s in val_starts])
# %%

class SDOLatentSeqDataset(torch.utils.data.Dataset):
    def __init__(self, latent_arrays, index_list, input_len, pred_len):
        # Each element of latent_arrays is (N_time_a, C, h, w) float32
        # torch.from_numpy shares memory — no copy
        self.latents = [torch.from_numpy(la).cuda() for la in latent_arrays]
        self.index = index_list
        self.in_len = input_len
        self.pred_len = pred_len
        self.win = input_len + pred_len

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        angle_id, s = self.index[i]
        clip = self.latents[angle_id][s : s + self.win]   # (W, C, h, w) — view
        return {
            "x": clip[:self.in_len],       # (T_in, C, h, w)
            "y": clip[self.in_len:],       # (T_pred, C, h, w)
            "angle_id": torch.tensor(angle_id, dtype=torch.long),
        }

# train_index and val_index from your previous pipeline still work,
# because they index by (angle_id, start_idx) into the per-angle arrays.
train_ds = SDOLatentSeqDataset(latent_arrays, train_index, INPUT_LEN, PRED_LEN)
val_ds   = SDOLatentSeqDataset(latent_arrays, val_index,   INPUT_LEN, PRED_LEN)
# %%
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)

# %%

class LSTMForecaster(nn.Module):
    def __init__(self, latent_dim=8192, hidden=512, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)  # maps hidden state → next latent

    def forward(self, z_in, z_target=None, pred_len=5):
        """
        z_in:     (B, T_in, latent_dim)
        z_target: (B, T_pred, latent_dim) — required for teacher forcing
        Returns:  (B, T_pred, latent_dim) predicted latents
        """
        B, T_in, D = z_in.shape

        # ---- Encoder phase: consume input sequence ----
        _, (h, c) = self.lstm(z_in)   # h, c: (n_layers, B, hidden)

        # ---- Decoder phase ----
        if z_target is not None:
            # Teacher forcing: the decoder inputs are the ground-truth previous frames.
            # At step t, we want to predict z_target[:, t]. Input at step t is the
            # previous ground-truth: for t=0 it's z_in[:, -1]; for t>0 it's z_target[:, t-1].
            last_input = z_in[:, -1:]                     # (B, 1, D)

            decoder_inputs = torch.cat(
                [last_input, z_target[:, :-1]], dim=1     # (B, T_pred, D)
            )
            # One parallel LSTM call across all T_pred steps:
            out, _ = self.lstm(decoder_inputs, (h, c))    # (B, T_pred, hidden)
            preds = self.head(out)                        # (B, T_pred, D)
            return preds

        else:
            # Autoregressive: feed predictions back in, one step at a time.
            preds = []
            x = z_in[:, -1:]                              # (B, 1, D)
            for _ in range(pred_len):
                out, (h, c) = self.lstm(x, (h, c))        # (B, 1, hidden)
                z_next = self.head(out)                   # (B, 1, D)
                preds.append(z_next)
                x = z_next                                # feed prediction back
            return torch.cat(preds, dim=1)                # (B, T_pred, D)

model2 = LSTMForecaster().cuda()

# %%
optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-4)
criterion = nn.L1Loss()

def train(model):
    l = []
    model.train()
    for batch in train_loader:
        x, y = batch['x'].cuda(), batch['y'].cuda()
        B, T_in, C, h, w = x.shape
        x_flat = x.reshape(B, T_in, C * h * w)        # (B, T_in, 8192)
        y_flat = y.reshape(B, y.shape[1], C * h * w)
        preds = model(x_flat, z_target=y_flat, pred_len=y.shape[1])
        loss = criterion(preds, y_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l.append(loss.item())
    return l

def valid(model):
    l = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['x'].cuda(), batch['y'].cuda()
            B, T_in, C, h, w = x.shape
            x_flat = x.reshape(B, T_in, C * h * w)        # (B, T_in, 8192)
            y_flat = y.reshape(B, y.shape[1], C * h * w)
            preds = model(x_flat, pred_len=y_flat.shape[1])
            loss = criterion(preds, y_flat)
            l.append(loss.item())
    return l

# %%

for i in range(50):
    train_l = train(model2)
    val_l = valid(model2)
    print(f'Epoch {i} done.')
    print(f'Train loss: {np.mean(train_l)}')
    print(f'Val loss: {np.mean(val_l)}')

torch.save(model2.state_dict(), '/home/oban/Desktop/Volga/MMI711/project/model_latent_lstm_conv1111111.pt')
