# %%
import os
import zarr;

import random
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
    train_starts = valid_starts[:-N_VAL_PER_ANGLE]
    val_starts   = valid_starts[-N_VAL_PER_ANGLE:]

    train_index.extend([(angle_id, s) for s in train_starts])
    val_index.extend  ([(angle_id, s) for s in val_starts])
print('3')
# ---------- Dataset ----------
class SDOSeqDataset(torch.utils.data.Dataset):
    def __init__(self, frame_arrays, index_list, input_len, pred_len):
        # Convert each angle's NumPy frames to a torch tensor (still on CPU).
        # torch.from_numpy shares memory with NumPy — no copy.
        self.frames = [torch.from_numpy(fa) for fa in frame_arrays]
        self.index = index_list           # list of (angle_id, start_idx)
        self.in_len = input_len
        self.pred_len = pred_len
        self.win = input_len + pred_len

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        angle_id, s = self.index[i]
        clip = self.frames[angle_id][s : s + self.win]   # (W, 64, 64) — view
        return {
            "x": clip[:self.in_len],
            "y": clip[self.in_len:],
            "angle_id": torch.tensor(angle_id, dtype=torch.long),
        }

train_ds = SDOSeqDataset(frame_arrays, train_index, INPUT_LEN, PRED_LEN)
val_ds   = SDOSeqDataset(frame_arrays, val_index,   INPUT_LEN, PRED_LEN)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)
print('Done')

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

cnn_ae = CNN_AE().eval().cuda()

state_dict = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_conv_ae2.pt', weights_only=False)
cnn_ae.load_state_dict(state_dict)

for param in cnn_ae.encoder.parameters():
    param.requires_grad = False

for param in cnn_ae.decoder.parameters():
    param.requires_grad = False

# %%
class LSTMForecaster(nn.Module):
    def __init__(self, latent_dim=8192, hidden=512, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)  # maps hidden state → next latent

    def forward(self, z_in, z_target=None, pred_len=5, teacher_forcing_ratio=0.5):
        """
        z_in:     (B, T_in, latent_dim)
        z_target: (B, T_pred, latent_dim) — required for teacher forcing
        Returns:  (B, T_pred, latent_dim) predicted latents
        """
        B, T_in, D = z_in.shape

        # ---- Encoder phase: consume input sequence ----
        _, (h, c) = self.lstm(z_in)   # h, c: (n_layers, B, hidden)

        # ---- Decoder phase ----
        if z_target is not None and random.random() < teacher_forcing_ratio:
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
state_dict2 = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_latent_lstm_conv.pt', weights_only=True)
model2.load_state_dict(state_dict2)

# %%

class CNN_LSTM_CNN(nn.Module):
    def __init__(self, hidden_dim=512, pred_steps=5):
        super(CNN_LSTM_CNN, self).__init__()
        self.pred_steps = pred_steps

        # 1. Encoder CNN: Processes frames individually
        self.encoder = cnn_ae.encoder
        # 2. LSTM: Processes the sequence of feature vectors
        # Input size is the flattened size of the last CNN layer
        self.lstm = model2

        # 4. Decoder CNN: Reconstructs vectors back to 64x64 frames
        self.decoder = cnn_ae.decoder

    def forward(self, x, y):
        # x shape: (Batch, 10, 64, 64)
        batch_size, seq_len, h, w = x.size()

        # Add channel dimension: (Batch, 10, 1, 64, 64)
        x = x.view(batch_size * seq_len, 1, h, w)
        y = y.view(batch_size * self.pred_steps, 1, h, w)

        # Encode all frames at once
        x_feat = self.encoder(x)
        y_feat = self.encoder(y)
        x_feat = x_feat.view(batch_size, seq_len, -1)
        y_feat = y_feat.view(batch_size, self.pred_steps, -1)

        preds = self.lstm(x_feat, z_target=y_feat)

        preds = preds.view(batch_size*self.pred_steps, 128, 8, 8)

        decoded = self.decoder(preds)
        decoded = decoded.view(batch_size, self.pred_steps, h, w)

        # Stack to (Batch, 5, 64, 64)
        return decoded

final_model = CNN_LSTM_CNN().cuda()
# %%

a = next(iter(train_loader))

print(a['y'].shape)
final_model(a['x'].cuda(),a['y'].cuda()).shape

# %%
from pytorch_msssim import ssim
import torch.nn.functional as F

def hybrid_loss(pred, target):
    l1   = F.l1_loss(pred, target)
    ssi = 1 - ssim(pred, target, data_range=1.0, size_average=True, win_size=11)
    return 0.2 * l1 + 0.8 * ssi

optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4)

def train(model):
    l = []
    model.train()
    for batch in train_loader:
        x, y = batch['x'].cuda(), batch['y'].cuda()
        #B, T_in, h, w = x.shape
        #x_flat = x.reshape(B, T_in, C * h * w)        # (B, T_in, 8192)
        #y_flat = y.reshape(B, y.shape[1], C * h * w)
        preds = model(x, y)
        loss = hybrid_loss(preds, y)
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
            #B, T_in, C, h, w = x.shape
            #x_flat = x.reshape(B, T_in, C * h * w)        # (B, T_in, 8192)
            #y_flat = y.reshape(B, y.shape[1], C * h * w)
            preds = model(x, y)
            loss = hybrid_loss(preds, y)
            l.append(loss.item())
    return l

# %%
for i in range(200):
    train_l = train(final_model)
    val_l = valid(final_model)
    if i%5 == 0:
        print(f'Epoch {i} done.')
        print(f'Train loss: {np.mean(train_l)}')
        print(f'Val loss: {np.mean(val_l)}')

torch.save(final_model.state_dict(), '/home/oban/Desktop/Volga/MMI711/project/model_lstm_finetuned.pt')
