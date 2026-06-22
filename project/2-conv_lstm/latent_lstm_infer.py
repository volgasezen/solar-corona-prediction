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

#train_ds = SDOSeqDataset(frame_arrays, train_index, INPUT_LEN, PRED_LEN)
val_ds   = SDOSeqDataset(frame_arrays, val_index,   INPUT_LEN, PRED_LEN)

#train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)
print('Done')
# %%
#a = next(iter(val_loader))
#
#a['angle_id']
#i = 6
#plt.imshow(a['y'][i,0,:,:].squeeze().numpy(), cmap='sdoaia171')
#plt.title(f'Angle: {ANGLES[a["angle_id"][i]]} deg.');
#plt.savefig(f'/home/oban/Desktop/Volga/MMI711/project/2-conv_lstm/{i}_y.jpg')
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

cnn_ae = CNN_AE().cuda()

state_dict = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_conv_ae2.pt', weights_only=False)
cnn_ae.load_state_dict(state_dict)

# %%
class LSTMForecaster(nn.Module):
    def __init__(self, latent_dim=8192, hidden=512, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)  # maps hidden state → next latent

    def forward(self, z_in, pred_len=5):
        """
        z_in:     (B, T_in, latent_dim)
        Returns:  (B, T_pred, latent_dim) predicted latents
        """
        #B, T_in, D_1, D_2, D_3 = z_in.shape
        #z_in = z_in.view(B, T_in, -1)

        # ---- Encoder phase: consume input sequence ----
        _, (h, c) = self.lstm(z_in)   # h, c: (n_layers, B, hidden)

        # ---- Decoder phase ----
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

    def forward(self, x):
        # x shape: (Batch, 10, 64, 64)
        batch_size, seq_len, h, w = x.size()

        # Add channel dimension: (Batch, 10, 1, 64, 64)
        x = x.view(batch_size * seq_len, 1, h, w)

        # Encode all frames at once
        features = self.encoder(x) # (Batch * 10, 8192)
        features = features.view(batch_size, seq_len, -1)

        #features = features.view(batch_size, seq_len, -1) # (Batch, 10, 8192)

        preds = self.lstm(features)
        preds = preds.view(batch_size, self.pred_steps, 128, 8, 8)
        preds = preds.permute(1, 0, 2, 3, 4)
        decoded = []
        for pred in preds:
            decoded.append(self.decoder(pred.squeeze()))

        # Stack to (Batch, 5, 64, 64)
        return torch.stack(decoded, dim=1).squeeze(2)

final_model = CNN_LSTM_CNN().cuda()
final_model.eval();
# %%


# %%
test_x = torch.stack([
    val_ds[i]['x'] for i in range(0, len(val_ds), 125)
])
test_y = torch.stack([
    val_ds[i]['y'] for i in range(0, len(val_ds), 125)
])
test_ang = torch.stack([
    val_ds[i]['angle_id'] for i in range(0, len(val_ds), 125)
])

# %%

with torch.no_grad():
    pred = final_model(test_x.cuda()).cpu().numpy() # (1, 5, 64, 64)
# %%
# Plot the 5th predicted frame vs the 5th actual frame
column_labels = ['GT (T=0)', 'GT (T=15)', 'Pred (T=15)']*2
# Plot the 5th predicted frame vs the 5th actual frame
fig, ax = plt.subplots(4, 6, figsize=(12, 8), dpi=300)
for i,ax in enumerate(ax.flatten()):
    if i < 6:
        ax.set_title(column_labels[i], fontsize=12, fontweight='bold')
    if i%3==0:
        ax.imshow(test_x[i//3, 0].squeeze(), cmap='sdoaia171')
    if i%3==1:
        ax.imshow(test_y[i//3, -1].squeeze(), cmap='sdoaia171')
    if i%3==2:
        ax.imshow(pred[i//3, -1].squeeze(), cmap='sdoaia171')
    ax.set_axis_off()
#ax[0].set_title("Model Prediction")
#ax[1].imshow(test_x.squeeze().numpy(), cmap='sdoaia171')
#ax[1].set_title("Ground Truth")
fig.tight_layout()
plt.show()
# %%

from matplotlib.animation import FFMpegWriter

chunk_size = 125
num_angles = 8

# 1. Setup the Figure ONCE
fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=300,layout='constrained')
axes_flat = axes.flatten()
ims_gt, ims_pred = [], []

# Initialize plots with empty arrays (we'll fill them in the loop)

for i,ax in enumerate(axes_flat):
    ax.set_title('.', fontsize=11, fontweight='bold')

for j in range(num_angles):
    ax_gt, ax_pred = axes_flat[j * 2], axes_flat[j * 2 + 1]
    ax_gt.set_axis_off()
    ax_pred.set_axis_off()

    # Create the image artists with a dummy 2D array
    im_gt = ax_gt.imshow(torch.zeros((64, 64)), cmap='sdoaia171', vmin=0, vmax=1)
    im_pred = ax_pred.imshow(torch.zeros((64, 64)), cmap='sdoaia171', vmin=0, vmax=1)

    ims_gt.append(im_gt)
    ims_pred.append(im_pred)

#fig.tight_layout()

# 2. Setup the Video Writer
writer = FFMpegWriter(fps=24, metadata=dict(artist='SolarModel'), bitrate=5000)
save_path = "project/2-conv_lstm/continuous_solar_validation.mp4"

print("Starting render. Press Ctrl+C at any time to stop and safely save the video.")

# 3. Stream frames directly into the video file
try:
    with writer.saving(fig, save_path, dpi=300):
        for offset in range(0,chunk_size,5):
            print(f"Rendering sequence {offset + 1}/{chunk_size}...")

            # --- Fetch Data & Predict for this sequence ---
            indices = [offset + (i * chunk_size) for i in range(num_angles)]
            test_x = torch.stack([val_ds[i]['x'] for i in indices])
            test_y = torch.stack([val_ds[i]['y'] for i in indices])

            with torch.no_grad():
                pred = final_model(test_x.cuda()).cpu().numpy() # (1, 5, 64, 64)

            # --- Loop through the 15 frames for this sequence ---
            for frame_idx in range(10, 15):
                for j in range(num_angles):
                    ax_gt, ax_pred = axes_flat[j * 2], axes_flat[j * 2 + 1]

                    future_idx = frame_idx - 10

                    gt_future = test_y[j, future_idx]
                    ims_gt[j].set_array(gt_future)
                    ax_gt.set_title(f"Angle {j} | Seq {offset} | GT (T={frame_idx})", fontsize=11, color='forestgreen', fontweight='bold')

                    pred_data = pred[j, future_idx]
                    ims_pred[j].set_array(pred_data)
                    ax_pred.set_title(f"Angle {j} | Seq {offset} | Pred (T={frame_idx})", fontsize=11, color='darkorange', fontweight='bold')

                # Snapshot the current figure and write it as a frame to the MP4
                writer.grab_frame()

except KeyboardInterrupt:
    print("\nRender cancelled by user! Finalizing video file safely...")

finally:
    # This ensures the figure memory is freed regardless of how the script ends
    plt.close(fig)
    print(f"Video saved successfully to: {save_path}")

# %%
import matplotlib.pyplot as plt
import numpy as np

# 1. Labels for the top row
column_labels = ['GT Δ (T=10-T=15)', 'Pred Δ (T=10-Pred)'] * 2

fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=300)
axes_flat = axes.flatten()

# 2. Set symmetric bounds to center white at 0
# If your errors are small, you can use a smaller value like 0.3 for more 'pop'
v_min, v_max = -0.05, 0.05

for i in range(16):
    ax = axes_flat[i]
    item_idx = i // 2  # Maps 0,1 to item 0; 2,3 to item 1, etc.

    # Calculate the error/difference
    if i % 2 == 0:
        # Ground Truth Difference: (T=10) - (T=15)
        error_data = test_x[item_idx, -1].squeeze() - test_y[item_idx, -1].squeeze()
    else:
        # Prediction Error: (T=10) - Predicted
        error_data = test_x[item_idx, -1].squeeze() - pred[item_idx, -1].squeeze()
    #error_data = np.abs(error_data)

    # 3. Use a diverging colormap ('RdBu_r', 'seismic', or 'coolwarm')
    # RdBu_r: Red = Positive error, Blue = Negative error, White = 0
    im = ax.imshow(error_data, cmap='RdBu_r', vmin=v_min, vmax=v_max)

    # Set titles for the first row only
    if i < 4:
        ax.set_title(column_labels[i], fontsize=12, fontweight='bold')

    # Optional: Label the rows by item index
    #if i % 4 == 0:
    #    ax.set_ylabel(f"Item {item_idx}", rotation=90, size='large', fontweight='bold')
    #    ax.set_axis_on() # Turn on only to show the ylabel
    #    ax.set_xticks([])
    #    ax.set_yticks([])
    #else:
    ax.set_axis_off()

# 4. Add a shared colorbar to explain the error magnitude
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Error Magnitude (Value Difference)')

plt.show()



# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
ax[0].imshow(pred[5, -1], cmap='sdoaia171')
ax[0].set_title("Model Prediction (T+5)")
ax[1].imshow(test_y[5, -1], cmap='sdoaia171')
ax[1].set_title("Ground Truth (T+5)")
ax[0].set_axis_off()
ax[1].set_axis_off()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_corona_patch_info(angle_deg, radius=206, size=64):
    """Calculates the center and the bounding box corner."""
    theta = np.radians(angle_deg)
    # cx is horizontal (cols), cy is vertical (rows)
    cx = 256 + radius * np.cos(theta)
    cy = 256 + radius * np.sin(theta)

    half = size / 2
    # Bounding box bottom-left corner for Matplotlib (x, y)
    corner_x = cx - half
    corner_y = cy - half

    return corner_x, corner_y, size

# --- Plotting ---
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(np.log1p(all_image[idx][0, :, :]), cmap='sdoaia171')

# Angles you want to visualize
angles = [0, 45, 90, 135, 180, 225, 270, 315]

for ang in angles:
    x, y, s = get_corona_patch_info(ang)

    # Create a Rectangle patch
    # (x, y) is the lower-left coordinate, then width, then height
    rect = patches.Rectangle((x, y), s, s, linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.8)

    # Add the patch to the Axes
    ax.add_patch(rect)

    # Optional: Add angle label
    ax.text(x, y-5, f'{ang}°', color='cyan', fontsize=10, fontweight='bold')

ax.set_axis_off()
plt.show()
# %%
df_time.sort_values('Time')['Time'].reset_index(drop=True)[10]
