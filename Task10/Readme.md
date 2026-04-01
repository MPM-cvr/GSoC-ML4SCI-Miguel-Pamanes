# Task X: Diffusion

In this section I will show you my procedure tu solve Task 10.

Task 10 ask the following

Complete the specific task 2 from the DeepFalcon test. Comment on potential ideas to extend this classical diffusion architecture to a quantum diffusion and sketch out the architecture in detail.

And this specific task from DeepFalcon is:

Specific Task 2 (if you are interested in “Diffusion Models for Fast Detector Simulation” project):
- Use a Diffusion Network model to represent the events in task 1. Please show a side-by side comparison of the original and reconstructed events and appropriate evaluation metric of your choice that estimates the difference between the two.

### Diffusion Networks

A Diffusion Model is a generative model that learns to create data by reversing a process of progressive degradation. Specifically, the algorithm adds Gaussian noise to an image step-by-step until the original content is completely indistinguishable. The model then learns to 'denoise' or undo this damage to reconstruct a clean image from pure noise.

<img width="1001" height="564" alt="image" src="https://github.com/user-attachments/assets/430a1108-511a-40e5-a69b-1f9c70f8135d" /><br>
(source: LeewayHertz. (s.f.). Diffusion Model [Imagen]. LeewayHertz.)

At their core, they are generative models that learn to create data by reversing a process of destruction. The most important part is
- Adding and Removing Noise: There are two steps
    - Forward Diffusion: We take a clear image and gradually add random Gaussian noise to it over many small steps. Eventually, the image is completely unrecognizable.
    - Reverse Diffusion: We train a neural network to look at a noisy image and predict exactly how much noise was added in that step. By subtracting that predicted noise, we get a slightly cleaner image.


### Code

First, I imported all the necessary packages

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import h5py
from skimage.metrics import structural_similarity as ssim
```

First, I began by processing the data. I loaded the variables X_jets, m0, pt, and y into memory. Finally, I printed the dimensions and data type of the primary variable to verify that the loading process was successful.
```Python
file_path = "/Users/miguelpamanes/Desktop/quark-gluon_data.hdf5"  

with h5py.File(file_path, "r") as f:
    print("Etiquetas disponibles:", list(f.keys()))
    
    X_jets = f["X_jets"][:]   
    m0 = f["m0"][:]           
    pt = f["pt"][:]         
    y = f["y"][:]           

print("Shape X_jets:", X_jets.shape)
print("Tipo:", X_jets.dtype)
```
That gave me the following

```Text
Etiquetas disponibles: ['X_jets', 'm0', 'pt', 'y']
Shape X_jets: (139306, 125, 125, 3)
Tipo: float32
```

I then sampled 5,000 collisions and scaled their values to a range between 0 and 1; this normalization prevents the model from being overwhelmed by large gradients or disparate scales. Finally, I reshaped the data to ensure the image dimensions are compatible with PyTorch's required input format.

```Python
X_jets = X_jets[:5000]
X_jets = X_jets / X_jets.max()

X_jets = np.transpose(X_jets, (0, 3, 1, 2))

print("Nuevo shape:", X_jets.shape)
```

So the new shape is

```Text
Nuevo shape: (5000, 3, 125, 125)
```


```Python
class JetDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

dataset = JetDataset(X_jets)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)



class DiffusionScheduler:
    def __init__(self, timesteps=200):
        self.timesteps = timesteps

        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,))

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)

        xt = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise
        return xt, noise



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb = torch.exp(
            torch.arange(half_dim, device=device) * (-np.log(10000) / (half_dim - 1))
        )

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb


class DiffusionCNN(nn.Module):
    def __init__(self, in_channels=3, time_dim=32):
        super().__init__()

        self.time_embed = TimeEmbedding(time_dim)

        self.conv1 = nn.Conv2d(in_channels + 1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, x, t):
        B, C, H, W = x.shape

        t_emb = self.time_embed(t)
        t_map = t_emb[:, 0].view(B, 1, 1, 1).repeat(1, 1, H, W)

        x = torch.cat([x, t_map], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)

        # Bottleneck
        self.middle = nn.Conv2d(128, 128, 3, padding=1)

        # Decoder
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, x, t):
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))

        m = F.relu(self.middle(d2))

        u1 = F.relu(self.up1(m))
        u1 = u1 + d1  # skip connection

        out = self.up2(u1)
        return out







def diffusion_loss(model, scheduler, x0):
    B = x0.shape[0]

    t = scheduler.sample_timesteps(B).to(x0.device)
    xt, noise = scheduler.add_noise(x0, t)

    noise_pred = model(xt, t)

    return F.mse_loss(noise_pred, noise)



def train(model, scheduler, dataloader, optimizer, device, epochs=5):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            x = batch.to(device)

            loss = diffusion_loss(model, scheduler, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")



@torch.no_grad()
def sample(model, scheduler, shape, device):
    x = torch.randn(shape).to(device)

    for t in reversed(range(scheduler.timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        alpha = scheduler.alpha[t]
        alpha_hat = scheduler.alpha_hat[t]
        beta = scheduler.beta[t]

        noise_pred = model(x, t_tensor)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred
        ) + torch.sqrt(beta) * noise

    return x



@torch.no_grad()
def reconstruct(model, scheduler, x0, device):
    model.eval()
    
    B = x0.shape[0]
    t = torch.full((B,), scheduler.timesteps - 1, device=device, dtype=torch.long)
    
    xt, _ = scheduler.add_noise(x0.to(device), t)

    x = xt
    for t_step in reversed(range(scheduler.timesteps)):
        t_tensor = torch.full((B,), t_step, device=device, dtype=torch.long)

        alpha = scheduler.alpha[t_step]
        alpha_hat = scheduler.alpha_hat[t_step]
        beta = scheduler.beta[t_step]

        noise_pred = model(x, t_tensor)

        if t_step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred
        ) + torch.sqrt(beta) * noise

    return x



device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionCNN(in_channels=3).to(device)
scheduler = DiffusionScheduler(timesteps = 50)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



train(model, scheduler, dataloader, optimizer, device, epochs=5)


generated = sample(model, scheduler, (16, 3, 125, 125), device)
generated = generated.cpu().numpy()
real = X_jets[:16]


reconstructed = reconstruct(
    model,
    scheduler,
    torch.tensor(real).float(),
    device
)
reconstructed = reconstructed.cpu().numpy()


def plot_comparison(real, generated, idx=0):
    r = real[idx].transpose(1,2,0)
    g = generated[idx].transpose(1,2,0)

    r = r / (r.max() + 1e-8)
    g = g / (g.max() + 1e-8)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(r)
    plt.title("Original Jet")

    plt.subplot(1,2,2)
    plt.imshow(g)
    plt.title("Reconstructed Jet")

    plt.show()

plot_comparison(real, reconstructed, 0)




def mse(x, y):
    return ((x - y)**2).mean()

print("MSE:", mse(real, reconstructed))


def compute_ssim(x, y):
    scores = []
    for i in range(len(x)):
        xi = x[i].transpose(1,2,0)
        yi = y[i].transpose(1,2,0)

        xi = np.clip(xi, 0, 1)
        yi = np.clip(yi, 0, 1)

        scores.append(ssim(xi, yi, channel_axis=2, data_range=1.0))
    
    return np.mean(scores)



plt.hist(real.flatten(), bins=50, alpha=0.5, label="Real")
plt.hist(generated.flatten(), bins=50, alpha=0.5, label="Generated")
plt.legend()
plt.title("Pixel Intensity Distribution")
plt.show()




```
