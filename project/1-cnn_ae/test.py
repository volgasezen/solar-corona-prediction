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

data = zarr.open("/home/oban/Desktop/Volga/MMI711/sdomlv2.zarr", mode='r')
headers = data.attrs.asdict()
all_image = da.from_zarr(data)

# %%
#print(np.percentile(np.log1p(sorted_patches.flatten()),100))
# %%
#m, s = train_x.mean(), train_x.std()
#plt.hist((np.log1p(sorted_patches.flatten())),bins=100);
#print(np.exp((np.max(train_ds)*s)+m)-1)
#print(np.max(sorted_patches))
#plt.yscale('log')
#plt.imshow(np.log1p(all_image[0,:,:]), cmap='sdoaia171')
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

def plot_sun_patch(id, angle):
    image = all_image[id,:,:]
    patch = get_corona_patch(image[np.newaxis,:,:], angle).squeeze()
    plt.imshow(np.log1p(patch), cmap='sdoaia171')

#plot_sun_patch(0, 0)

# %%
t_obs = np.array(headers["T_OBS"])
df_time = pd.DataFrame(t_obs, index=np.arange(np.shape(t_obs)[0]), columns=["Time"])
df_time["Time"] = pd.to_datetime(df_time["Time"])

idx = df_time.sort_values('Time').index

n_total = len(idx)
n_val = 125  # per angle
val_idx, train_idx = idx[-n_val:], idx[:-n_val]

train_patches = da.concatenate(
    [get_corona_patch(all_image, a)[train_idx] for a in range(0, 360, 45)], axis=0
).compute()
val_patches = da.concatenate(
    [get_corona_patch(all_image, a)[val_idx] for a in range(0, 360, 45)], axis=0
).compute()

#sorted_patches = da.concatenate([get_corona_patch(all_image, x)[idx] for x in range(0,360,45)],axis=0)

# %%

#log_train = np.log1p(train_patches)
#log_min = np.percentile(log_train, 0.5)
#log_max = np.percentile(log_train, 99.5)
#np.save('/home/oban/Desktop/Volga/MMI711/project/train_min_max.npy',np.array([log_min, log_max]))
log_min, log_max = np.load('/home/oban/Desktop/Volga/MMI711/project/train_min_max.npy')
scale = log_max - log_min

train_x = np.clip((np.log1p(train_patches) - log_min) / scale, 0, 1)

val_x = np.clip((np.log1p(val_patches) - log_min) / scale, 0, 1)

train_loader = torch.utils.data.DataLoader(train_x, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_x, batch_size=128)

# %%

next(iter(val_loader)).shape

# %%
import torch
import torch.nn as nn

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
# %%
#print(np.max(sorted_patches))
model = CNN_AE().cuda()
#a=next(iter(train_loader)).cuda()
#torch.max((torch.exp((torch.tensor(train_x)*s)+m)-1))
#model(a.unsqueeze(1)).shape
# %%
from pytorch_msssim import ssim
import torch.nn.functional as F

def hybrid_loss(pred, target):
    l1   = F.l1_loss(pred, target)
    ssi = 1 - ssim(pred, target, data_range=1.0, size_average=True, win_size=11)
    return 0.2 * l1 + 0.8 * ssi

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def train(model):
    model.train()
    l = []
    for x in train_loader:
        x = x.unsqueeze(1).cuda()
        preds = model(x)
        loss = hybrid_loss(preds, x)
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
        for x in val_loader:
            x = x.unsqueeze(1).cuda()
            preds = model(x)
            loss = hybrid_loss(preds, x)
            l.append(loss.item())
    return l

# %%

for i in range(200):
    train_l = train(model)
    val_l = valid(model)
    if i%5 == 0:
        print(f'Epoch {i} done.')
        print(f'Train loss: {np.mean(train_l)}')
        print(f'Val loss: {np.mean(val_l)}')

torch.save(model.state_dict(), '/home/oban/Desktop/Volga/MMI711/project/model_conv_ae2.pt')
# %%
state_dict = torch.load('project/model_conv_ae2.pt', weights_only=False)
model.load_state_dict(state_dict)
# %%
#import torchvision

#torchvision.transforms.functional.rotate(test_x, 90).shape

# %%
#val_ds[0:1][0].shape
## %%
model.eval()

test_x = torch.stack([
    torch.tensor(val_x[i:i+1,:,:]) for i in range(0, len(val_x), 125)
])

#print(test_x.shape)
#test_x = torchvision.transforms.functional.rotate(test_x, 90)
#test_x = torch.ones_like(test_x)
#test_y = torchvision.transforms.functional.rotate(test_y, 90)
with torch.no_grad():
    pred = model(test_x.cuda()).cpu().numpy() # (1, 5, 64, 64)

column_labels = ['Ground Truth', 'Predicted']*2
# Plot the 5th predicted frame vs the 5th actual frame
fig, ax = plt.subplots(4, 4, figsize=(10, 10), dpi=300)
for i,ax in enumerate(ax.flatten()):
    if i < 4:
        ax.set_title(column_labels[i], fontsize=12, fontweight='bold')
    if i%2==1:
        ax.imshow(pred[i//2].squeeze(), cmap='sdoaia171')
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
import matplotlib.pyplot as plt
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

        plt.title('Training and Validation Loss')
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
plot_learning_curves('/home/oban/Desktop/Volga/MMI711/project/1-cnn_ae/losses.txt')
# %%
a = torch.rand((1,10,10,10))
b = torch.rand((1,10,10,10))
#print(a.max())
ssim(a, b, data_range=1)

# %%
testt_x = torch.tensor(val_x.reshape(8,-1,64,64)).cuda()

ssis = []
maes = []
for i in range(8):
    tested = testt_x[i,:,:,:].squeeze().unsqueeze(1)
    with torch.no_grad():
        pred = model(tested)
        ssi = 1 - ssim(pred, tested, data_range=1.0, size_average=False, win_size=11)
        mae = F.l1_loss(pred, tested, reduction='none')
        ssis.append(ssi.tolist())
        maes.append(mae.tolist())

len(maes[0])

# %%
maes[0][0][0][0]
# %%
import matplotlib.pyplot as plt
import numpy as np

# Define the degrees to match your previous plot
degrees = [0, 45, 90, 135, 180, 225, 270, 315]

# Choose a subtle, professional colormap (e.g., 'viridis', 'plasma', or 'coolwarm')
cmap = plt.get_cmap('viridis', 8)

fig, ax = plt.subplots(8, 1, figsize=(5, 10), sharex=True, sharey=True, dpi=300)

for i in range(8):
    # Select color from the palette

    # Plot the histogram
    ax[i].hist(ssis[i], bins=30, alpha=0.6, color='k', edgecolor='white', linewidth=0.5)

    # Add the degree text to the right side of the plot for clarity
    # transform=ax[i].transAxes allows us to use 0-1 coordinates for placement
    ax[i].text(1.02, 0.5, f'{degrees[i]}°', transform=ax[i].transAxes,
               va='center', ha='left', fontsize=12, fontweight='bold', color='k')

    # Clean up the look
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

    # Add a subtle grid for the x-axis only
    ax[i].grid(axis='x', linestyle='--', alpha=0.3)

# Add shared labels
fig.supxlabel('1 - SSIM (Error)', fontsize=14)
fig.supylabel('Frequency', fontsize=14)
fig.suptitle('SSIM Error Distribution per Angle', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
# %%
torch.save(model.state_dict(), '/home/oban/Desktop/Volga/MMI711/project/model_conv_ae3.pt')
