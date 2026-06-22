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
# %%

data = zarr.open("sdomlv2.zarr", mode='r')
headers = data.attrs.asdict()
# %%

all_image = da.from_zarr(data)
#image=all_image[6000,:,:]
#plt.figure(figsize=(10,10))
#colormap = plt.get_cmap('sdoaia171')
#plt.imshow(np.log1p(image),cmap=colormap)
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

#plt.imshow(get_corona_patch(image[np.newaxis,:,:], 220).squeeze(), origin='lower',vmin=10,vmax=1000,cmap=colormap)
# %%
t_obs = np.array(headers["T_OBS"])
df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
df_time["Time"] = pd.to_datetime(df_time["Time"])

idx = df_time.sort_values('Time').index
patches = get_corona_patch(all_image, 220)
sorted_patches = patches[idx]

# %%
from numpy.lib.stride_tricks import sliding_window_view

def get_windows_fast(data, window_size, pred_size):
    # This creates a view of the array, not a copy (extremely fast)
    # Shape will be (N, window_size, H, W)
    full_window = window_size + pred_size
    windows = sliding_window_view(data, window_shape=(full_window, 64, 64), axis=(0, 1, 2))

    # Remove the extra dimensions created by sliding_window_view
    windows = windows.squeeze()

    x = windows[:, :window_size]
    y = windows[:, window_size:]
    return x, y

x, y = get_windows_fast(sorted_patches, 10, 5)

#windows = get_windows(patches,diffs,10,5)
# %%
x = x.compute()
y = y.compute()
print(x.shape, y.shape)

# %%
diffs = np.array([0]+[x.total_seconds() for x in np.diff(sorted(df_time['Time']))])
stops = np.where(diffs > 1080)[0]

num_windows = len(x)
valid_mask = np.ones(num_windows, dtype=bool)

for s in stops:
    start_bad = max(0, s - (10+5+1))
    end_bad = min(num_windows, s + 1)
    valid_mask[start_bad:end_bad] = False

x, y = x[valid_mask,:,:,:], y[valid_mask,:,:,:]
# %%
class SDOIAIDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, mean, std):
        # Store as tensors
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Normalize on the fly
        x_norm = (self.x[idx] - self.mean) / self.std
        y_norm = (self.y[idx] - self.mean) / self.std
        return x_norm, y_norm

train_x, train_y = np.log1p(x[:-1000,:,:,:]), np.log1p(y[:-1000,:,:,:])
val_x, val_y = np.log1p(x[-985:,:,:,:]), np.log1p(y[-985:,:,:,:])

m, s = train_x.mean(), train_x.std()

train_ds = SDOIAIDataset(train_x, train_y, m, s)
val_ds = SDOIAIDataset(val_x, val_y, m, s)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)

# %%
import torch
import torch.nn as nn

class CNN_LSTM_CNN(nn.Module):
    def __init__(self, hidden_dim=512, pred_steps=5):
        super(CNN_LSTM_CNN, self).__init__()
        self.pred_steps = pred_steps

        # 1. Encoder CNN: Processes frames individually
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Flatten() # Result: 128 * 8 * 8 = 8192
        )
        # 2. LSTM: Processes the sequence of feature vectors
        # Input size is the flattened size of the last CNN layer
        self.lstm = nn.LSTM(input_size=8192, hidden_size=hidden_dim,
                            num_layers=2, batch_first=True)

        # 3. Linear bottleneck to bridge LSTM to Decoder
        self.to_decoder = nn.Linear(hidden_dim, 8192)

        # 4. Decoder CNN: Reconstructs vectors back to 64x64 frames
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 32 -> 64
        )

    def forward(self, x):
        # x shape: (Batch, 10, 64, 64)
        batch_size, seq_len, h, w = x.size()

        # Add channel dimension: (Batch, 10, 1, 64, 64)
        x = x.view(batch_size * seq_len, 1, h, w)

        # Encode all frames at once
        features = self.encoder(x) # (Batch * 10, 8192)
        features = features.view(batch_size, seq_len, -1) # (Batch, 10, 8192)

        # LSTM pass
        out, (hn, cn) = self.lstm(features)

        # We want to predict the next 5 steps.
        # A common trick is to use the last hidden state (hn)
        # or the last sequence output to generate future frames.
        last_output = out[:, -1, :] # Take the last time step: (Batch, hidden_dim)

        predictions = []
        curr_feat = last_output

        for i in range(self.pred_steps):
            # Decode the current state into a frame
            decoded_feat = self.to_decoder(curr_feat)
            frame = self.decoder(decoded_feat)
            predictions.append(frame)

            # If we were doing auto-regressive, we'd re-encode 'frame'
            # and pass back through LSTM here. For a fixed 5-step,
            # we can also use a multi-head linear output.

        # Stack to (Batch, 5, 64, 64)
        return torch.stack(predictions, dim=1).squeeze(2)

model = CNN_LSTM_CNN().cuda()
# %%
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

def train(model):
    model.train()
    l = []
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        preds = model(x)
        loss = criterion(preds, y)
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    l.append(loss.item())
    return l

def valid(model):
    l = []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            preds = model(x)
            loss = criterion(preds, y)
            l.append(loss.item())
    return l

# %%

for i in range(1000):
    train_l = train(model)
    val_l = valid(model)
    if i%5 == 0:
        print(f'Epoch {i} done.')
        print(f'Train loss: {np.mean(train_l)}')
        print(f'Val loss: {np.mean(val_l)}')

torch.save(model.state_dict(), 'project/model.pt')
# %%
state_dict = torch.load('/home/oban/Desktop/Volga/MMI711/project/0-cnn_lstm_cnn/model.pt', weights_only=False)
model.load_state_dict(state_dict)
model
# %%
import torchvision

torchvision.transforms.functional.rotate(test_x, 90).shape

# %%
val_ds[0:1][0].shape
# %%
i = 1000
model.eval()
test_x, test_y = train_ds[i:i+1]
#test_x = torchvision.transforms.functional.rotate(test_x, 90)
#test_x = torch.ones_like(test_x)
#test_y = torchvision.transforms.functional.rotate(test_y, 90)
with torch.no_grad():
    pred = model(test_x.cuda()).cpu().numpy() # (1, 5, 64, 64)
    target = test_y.cuda()[0].cpu().numpy()        # (5, 64, 64)

# Plot the 5th predicted frame vs the 5th actual frame
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
ax[0].imshow(pred[0, -1], cmap='sdoaia171')
ax[0].set_title("Model Prediction (T+5)")
ax[1].imshow(target[-1], cmap='sdoaia171')
ax[1].set_title("Ground Truth (T+5)")
ax[0].set_axis_off()
ax[1].set_axis_off()
plt.show()

# %%
torch.diff(test_y.squeeze(),axis=0).shape

plt.imshow(torch.diff(test_x.squeeze(),axis=0)[-1,:,:], vmax=2, vmin=-2)
# %%
plt.imshow(test_x.squeeze()[-1,:,:]-test_y.squeeze()[-1,:,:])
torch.max(test_x.squeeze()[-1,:,:]-test_y.squeeze()[-1,:,:])
torch.min(test_x.squeeze()[-1,:,:]-test_y.squeeze()[-1,:,:])
s
