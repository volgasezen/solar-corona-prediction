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
import re

def plot_learning_curves(file_path):
    epochs = []
    train_losses = []
    val_losses = []

    # Regular expressions to find the numbers
    epoch_pattern = re.compile(r"Epoch (\d+) done")
    train_pattern = re.compile(r"Train loss: ([\d.]+)")
    val_pattern = re.compile(r"Val loss: ([\d.]+)")

    try:
        with open(file_path, 'r') as f:
            content = f.read()

            # Find all matches
            epochs = [int(e) for e in epoch_pattern.findall(content)]
            train_losses = [float(t) for t in train_pattern.findall(content)]
            val_losses = [float(v) for v in val_pattern.findall(content)]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss')#, marker='o', markersize=4)
        plt.plot(epochs, val_losses, label='Val Loss')#, marker='o', markersize=4)

        plt.title('Training and Validation Loss (Latent LSTM)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.grid(True, linestyle='--', alpha=0.7)

        # Log scale can be helpful if the initial loss is much higher than final
        plt.yscale('log')

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

# Usage: Change 'losses.txt' to your actual filename
plot_learning_curves('/home/oban/Desktop/Volga/MMI711/project/3-lstm2/log4.txt')


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
class CNN_AE(nn.Module):
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

cnn_ae = CNN_AE().cuda()

state_dict = torch.load('/home/oban/Desktop/Volga/MMI711/project/3-lstm2/model_conv_ae3.pt', weights_only=False)
cnn_ae.load_state_dict(state_dict)

# %%
import sys
sys.path.append('/home/oban/Desktop/Volga/MMI711/project/3-lstm2')
from convlstm import ConvLSTM

class ConvLSTMForecaster(nn.Module):
    def __init__(self, hidden_ch=16, n_layers=3):
        super().__init__()
        self.convlstm = ConvLSTM(hidden_ch, [16,32,16], [(3,3)]*3, num_layers=n_layers, batch_first=True)

    def forward(self, z_in, z_target=None, pred_len=5, teacher_forcing_ratio=0.5):
        """
        z_in:     (B, T_in, latent_dim)
        z_target: (B, T_pred, latent_dim) — required for teacher forcing
        Returns:  (B, T_pred, latent_dim) predicted latents
        """
        B, T_in, in_ch, A, B = z_in.shape

        # ---- Encoder phase: consume input sequence ----
        _, hidden_list = self.convlstm(z_in)   # h, c: (n_layers, B, hidden)

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
            out, _ = self.convlstm(decoder_inputs, hidden_list)    # (B, T_pred, hidden)
            return out[-1]

        else:
            # Autoregressive: feed predictions back in, one step at a time.
            preds = []
            x = z_in[:, -1:]                              # (B, 1, D)
            for _ in range(pred_len):
                out, hidden_list = self.convlstm(x, hidden_list)        # (B, 1, hidden)
                preds.append(out[-1])
                x = out[-1]                                # feed prediction back
            return torch.cat(preds, dim=1)                # (B, T_pred, D)

class ConvLSTMForecaster2(nn.Module):
    def __init__(self, hidden_ch=16, n_layers=3):
        super().__init__()
        self.convlstm = ConvLSTM(hidden_ch, [16,32,16], [(3,3)]*3, num_layers=n_layers, batch_first=True)

    def forward(self, z_in, z_target=None, pred_len=5, teacher_forcing_ratio=0.5):
        """
        z_in:     (B, T_in, latent_dim)
        z_target: (B, T_pred, latent_dim) — required for teacher forcing
        Returns:  (B, T_pred, latent_dim) predicted latents
        """
        B, T_in, in_ch, A, B = z_in.shape

        # ---- Encoder phase: consume input sequence ----
        _, hidden_list = self.convlstm(z_in)   # h, c: (n_layers, B, hidden)

        # ---- Decoder phase ----
        if z_target is not None and random.random() < teacher_forcing_ratio:
            # Teacher forcing: the decoder inputs are the ground-truth previous frames.
            # At step t, we want to predict z_target[:, t]. Input at step t is the
            # previous ground-truth: for t=0 it's z_in[:, -1]; for t>0 it's z_target[:, t-1].
            last_input = z_in[:, -1:]
            decoder_inputs = torch.cat([last_input, z_target[:, :-1]], dim=1)

            # The LSTM now outputs VELOCITIES (the change between frames)
            velocities_list, _ = self.convlstm(decoder_inputs, hidden_list)
            velocities = velocities_list[-1]

            # THE TRICK: Frame_Next = Frame_Current + Velocity
            absolute_preds = decoder_inputs + velocities

            return absolute_preds

        else:
            # Autoregressive: feed predictions back in, one step at a time.
            preds = []
            x = z_in[:, -1:]

            for _ in range(pred_len):
                # 1. The LSTM outputs the VELOCITY
                velocities_list, hidden_state = self.convlstm(x, hidden_list)
                velocity = velocities_list[-1]

                # 2. THE TRICK: Add the velocity to the current frame
                out_tensor = x + velocity

                preds.append(out_tensor)

                # 3. Feed the absolute predicted frame back in for the next step
                x = out_tensor

            return torch.cat(preds, dim=1)

lstm = ConvLSTMForecaster2().cuda()

#state_dict2 = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_latent_convlstm.pt', weights_only=False)
#lstm.load_state_dict(state_dict2)

# %%
class CNN_LSTM_CNN(nn.Module):
    def __init__(self, hidden_dim=512, pred_steps=5):
        super(CNN_LSTM_CNN, self).__init__()
        self.pred_steps = pred_steps

        # 1. Encoder CNN: Processes frames individually
        self.encoder = cnn_ae.encoder
        # 2. LSTM: Processes the sequence of feature vectors
        # Input size is the flattened size of the last CNN layer
        self.lstm = lstm

        # 4. Decoder CNN: Reconstructs vectors back to 64x64 frames
        self.decoder = cnn_ae.decoder

    def forward(self, x, pred_steps):
        # x shape: (Batch, 10, 64, 64)
        batch_size, seq_len, h, w = x.size()

        # Add channel dimension: (Batch, 10, 1, 64, 64)
        x = x.view(batch_size * seq_len, 1, h, w)

        # Encode all frames at once
        features = self.encoder(x)

        features = features.view(batch_size, seq_len, 16, 8, 8) # (Batch, 10, 8192)

        preds = self.lstm(features, pred_len = pred_steps)
        #print(preds.shape)
        #preds = preds.view(batch_size, self.pred_steps, 16, 8, 8)
        preds = preds.permute(1, 0, 2, 3, 4)
        #print(preds.shape)
        decoded = []
        for pred in preds:
            decoded.append(self.decoder(pred.squeeze()))

        # Stack to (Batch, 5, 64, 64)
        return torch.stack(decoded, dim=1).squeeze(2)

final_model = CNN_LSTM_CNN().cuda()
#state_dict3 = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_convlstm_finetuned.pt', weights_only=False)
state_dict3 = torch.load('/home/oban/Desktop/Volga/MMI711/project/model_convlstm_finetuned2.pt', weights_only=False)
final_model.load_state_dict(state_dict3)
final_model.eval()
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
pred.shape


# %%

with torch.no_grad():
    pred = final_model(test_x.cuda()).cpu().numpy() # (1, 5, 64, 64)

# %%

B, L, H, W = test_x.shape
test_x = test_x.view(B*L, 1, H, W)

with torch.no_grad():
    x_recon = cnn_ae(test_x.cuda())[0].cpu().numpy() # (1, 5, 64, 64)

# %%
column_labels = ['Ground Truth', 'Predicted']*2
# Plot the 5th predicted frame vs the 5th actual frame
fig, ax = plt.subplots(4, 4, figsize=(10, 10), dpi=300)
for i,ax in enumerate(ax.flatten()):
    if i < 4:
        ax.set_title(column_labels[i], fontsize=12, fontweight='bold')
    if i%2==1:
        ax.imshow(x_recon[i//2].squeeze(), cmap='sdoaia171')
        #ax.set_title("Model Prediction")
    else:
        ax.imshow(test_x[i//2,:,:].squeeze().numpy(), cmap='sdoaia171')
        #ax.set_title("Ground Truth")
    ax.set_axis_off()
#ax[0].set_title("Model Prediction")
#ax[1].imshow(test_x.squeeze().numpy(), cmap='sdoaia171')
#ax[1].set_title("Ground Truth")
plt.show()


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
save_path = "project/2-conv_lstm/continuous_solar_validation_v4.mp4"

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

def evaluate_per_frame_loss(model, val_loader, pred_len=5):
    model.eval()

    # Store the average loss for step 0, 1, 2, 3, 4
    #per_frame_mse = torch.zeros(pred_len).cuda()
    #total_batches = 0
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].cuda() # (B, 10, C, H, W)
            y = batch['y'].cuda() # (B, 5, C, H, W)

            # Run your autoregressive prediction
            preds = model(x)      # (B, 5, C, H, W)

            # Calculate MSE independently for EACH frame in the prediction window
            for t in range(pred_len):
                # Calculate MSE between the t-th predicted frame and t-th GT frame
                mse_t = F.mse_loss(preds[:, t], y[:, t]).item()
                losses.append(mse_t)

            #total_batches += 1

    # Average across all batches
    #per_frame_mse = (per_frame_mse / total_batches).cpu().numpy()

    # Plot the degradation curve
    plt.figure(figsize=(8, 5))
    plt.plot(torch.tensor(losses).reshape(len(val_loader), 5), marker='o', color='red', linewidth=2)
    plt.title('Autoregressive Error Accumulation per Frame')
    plt.xlabel('Prediction Step (Frames into the future)')
    plt.ylabel('Mean Squared Error (MSE)')
    #plt.xticks(range(1, pred_len + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Run it
evaluate_per_frame_loss(final_model, val_loader, pred_len=5)

# %%
from matplotlib.animation import FFMpegWriter
import torch
import matplotlib.pyplot as plt

chunk_size = 125
num_angles = 8

# 1. Setup the Figure ONCE
fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=300, layout='constrained')
axes_flat = axes.flatten()
ims_gt, ims_pred = [], []

for i, ax in enumerate(axes_flat):
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

# 2. Predict the ENTIRE sequence at once
print("Generating 125-frame deep autoregressive rollout...")

# IMPORTANT: You must tell your model to predict the full chunk size before running!
final_model.pred_len = chunk_size
final_model.eval()

# Fetch ONLY the initial 10 frames (offset = 0) to kick off the prediction
start_indices = [0 + (i * chunk_size) for i in range(num_angles)]
initial_x = torch.stack([val_ds[i]['x'] for i in start_indices]).cuda()

with torch.no_grad():
    # pred shape will be: (8, 125, 64, 64)
    pred = final_model(initial_x, chunk_size).cpu().numpy()
print(pred.shape)
# 3. Setup the Video Writer
writer = FFMpegWriter(fps=24, metadata=dict(artist='SolarModel'), bitrate=5000)
save_path = "project/2-conv_lstm/continuous_solar_validation_v4_deep_rollout.mp4"

print("Starting render. Press Ctrl+C at any time to stop and safely save the video.")

# 4. Stream the predicted frames to the video
try:
    with writer.saving(fig, save_path, dpi=300):

        # Loop strictly through the temporal dimension we just predicted
        for t in range(chunk_size):
            if t % 10 == 0:
                print(f"Rendering frame {t + 1}/{chunk_size}...")

            # Fetch the matching Ground Truth for time step t
            # Because val_ds[0]['y'] starts at T=10, val_ds[t]['y'][0] perfectly equals T=10+t
            gt_indices = [t + (i * chunk_size) for i in range(num_angles)]
            gt_y_t = torch.stack([val_ds[i]['y'][0] for i in gt_indices]).numpy()

            for j in range(num_angles):
                ax_gt, ax_pred = axes_flat[j * 2], axes_flat[j * 2 + 1]

                # Update GT
                ims_gt[j].set_array(gt_y_t[j])
                ax_gt.set_title(f"Angle {j} | GT (T={10 + t})", fontsize=11, color='forestgreen', fontweight='bold')

                # Update Pred
                ims_pred[j].set_array(pred[j, t])
                ax_pred.set_title(f"Angle {j} | Pred (T={10 + t})", fontsize=11, color='darkorange', fontweight='bold')

            # Snapshot the current figure and write it
            writer.grab_frame()

except KeyboardInterrupt:
    print("\nRender cancelled by user! Finalizing video file safely...")

finally:
    plt.close(fig)
    print(f"Video saved successfully to: {save_path}")
