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

Next, I converted the matrices into native PyTorch Tensors. I then implemented a DataLoader to act as an automated pipeline; this handles shuffling the dataset and delivering the data to the neural network in mini-batches of 8 images at a time.

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
```

I then implemented the DiffusionScheduler class to manage the Forward Process. This involves taking the clean data and progressively adding Gaussian noise until the original structure is completely destroyed, leaving only pure static.

```Python
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
```

I then developed the TimeEmbedding class. For the neural network to effectively denoise the particle collision images, it must be aware of the specific time step t. This class transforms that scalar value t into a high-dimensional 'signature' using sinusoidal embeddings, allowing the model to distinguish between different stages of the diffusion process.


```Python
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
```

Then I Designed two different architectures to solve the same problem.

### Diffusion CNN
I implemented a 4-layer 'flat' CNN where the image resolution remains constant throughout the process. The model integrates temporal context by concatenating the time embedding as an extra channel. However, because it lacks downsampling layers, the network has a limited receptive field; this forces it to focus on local pixel details rather than capturing the global structure of the particle collision

```Python
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
```
### U-Net

Also I designed and implemented a second architechture, an U-Net architecture as the core model, integrating linear adapters to inject time-awareness into every block. By utilizing skip connections, the network can effectively reconstruct images from pure noise, successfully balancing the recovery of global structure with the preservation of fine particle details.

```Python
class UNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=32):
        super().__init__()

        self.time_embed = TimeEmbedding(time_dim)

        self.te_down = nn.Linear(time_dim, 64)
        self.te_mid = nn.Linear(time_dim, 128)
        self.te_up = nn.Linear(time_dim, 64)

        # Encoder
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)

        # Bottleneck
        self.middle = nn.Conv2d(128, 128, 3, padding=1)

        # Decoder
        self.up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        d1 = F.relu(self.down1(x))
        t_d1 = self.te_down(t_emb).view(-1, 64, 1, 1) 
        d1 = d1 + t_d1

        d2 = F.relu(self.down2(d1))

        m = F.relu(self.middle(d2))
        t_m = self.te_mid(t_emb).view(-1, 128, 1, 1)
        m = m + t_m

        u1 = F.relu(self.up1(m))
        t_u1 = self.te_up(t_emb).view(-1, 64, 1, 1)
        u1 = u1 + t_u1

        u1 = u1 + d1  
        out = self.up2(u1)
        return out
```


Then I designed the diffusion_loss function. This function constitutes the optimization core of the generative model. Its fundamental purpose is to quantify the discrepancy between the actual stochastic noise injected into the data and the prediction of said noise made by the neural network (be it a CNN or U-Net architecture). By minimizing this error, the network learns to reverse the signal corruption process.

```Python

def diffusion_loss(model, scheduler, x0):
    B = x0.shape[0]

    t = scheduler.sample_timesteps(B).to(x0.device)
    xt, noise = scheduler.add_noise(x0, t)

    noise_pred = model(xt, t)

    return F.mse_loss(noise_pred, noise)
```

In this code block, the main model optimization routine was implemented through a continuous training cycle. Over multiple eras, the algorithm extracts the information into small subsets, transfers the particle tensors to the corresponding hardware and calculates the Mean Square Error using the diffusion loss function. From this value, the system uses the backpropagation algorithm and an optimizer to adjust iteratively and mathematically the internal weights of the neural network, ending each era with a report of the average error.


```Python
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
```

Then the data generation process is executed through reverse diffusion. The algorithm begins by creating a pure Gaussian noise tensor and, through an iterative cycle that goes back in time step by step, uses the previously trained neural network to predict and subtract the noise present in each stage. Following the formal Denoising Diffusion Probabilistic Models (DDPM) equations, the model progressively cleans the matrix by injecting small amounts of stabilizing noise, until, when reaching the zero time step, the initial chaos converges into a completely new, coherent and high-fidelity particle collision simulation.

```Python
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
```


In this code block I implemented a validation routine to evaluate the retention capacity and geometric fidelity of the neural network. Unlike free generation, this function takes a set of real collision matrices, injects them with the maximum level of Gaussian noise stipulated by the planner until the signal is obliterated, and subsequently uses the reverse diffusion iterative process of the DDPM models to try to recover the original structure.


```Python
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
```

In the final implementation phase, I instantiated the network architecture and noise planner with 100 time steps, assigning the Adam optimizer with a learning rate of 1e-4 to ensure stable convergence. Finally, I orchestrated all these dependencies in the main cycle to train the model on particle collision simulations for 20 continuous periods, thus consolidating the machine learning process.

This first example is for DifussionCNN

```Python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionCNN(in_channels=3).to(device)
scheduler = DiffusionScheduler(timesteps = 50)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train(model, scheduler, dataloader, optimizer, device, epochs=5)
```

That gave me the following results 
```Text
Epoch 1: Loss = 0.2097
Epoch 2: Loss = 0.0528
Epoch 3: Loss = 0.0372
Epoch 4: Loss = 0.0314
Epoch 5: Loss = 0.0277
Epoch 6: Loss = 0.0251
Epoch 7: Loss = 0.0223
Epoch 8: Loss = 0.0201
Epoch 9: Loss = 0.0195
Epoch 10: Loss = 0.0180
Epoch 11: Loss = 0.0165
Epoch 12: Loss = 0.0155
Epoch 13: Loss = 0.0154
Epoch 14: Loss = 0.0138
Epoch 15: Loss = 0.0133
Epoch 16: Loss = 0.0125
Epoch 17: Loss = 0.0129
Epoch 18: Loss = 0.0115
Epoch 19: Loss = 0.0116
Epoch 20: Loss = 0.0111
```

And the following code is for the U-NET

```Python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channels=3).to(device)
scheduler = DiffusionScheduler(timesteps = 50)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train(model, scheduler, dataloader, optimizer, device, epochs=5)
```
That gave the following results
```Text

```


To evaluate the model, I generated a batch of 16 synthetic simulations from scratch and processed another group of 16 actual collisions through the reconstruction function. Finally, I transferred all the resulting tensors from the GPU to the CPU and converted them to NumPy fixes, preparing the data in the ideal format to visually and statistically compare real, reconstructed and artificially generated samples.


```Python
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
```


To qualitatively inspect the results, I designed a visualization function using the Matplotlib library. The algorithm takes the PyTorch tensors, reorganizes their spatial dimensions to the compatible Matplotlib format and normalizes the pixel intensities to the range of [0.1]. Finally, the code renders both simulations in an adjacent way, facilitating direct visual comparison of the particle distribution.

```Python
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
```
That gave me the next comparission for CNN Difussion Model
<img width="830" height="416" alt="Captura de pantalla 2026-04-03 a la(s) 9 19 43 p m" src="https://github.com/user-attachments/assets/a4b6b946-97ce-4a0f-a235-90663e4650ff" />

And gave this for the U-Net model



To complement the visual inspection with an objective evaluation metric, I implemented the calculation of the Mean Square Error (MSE).
```Python
def mse(x, y):
    return ((x - y)**2).mean()

print("MSE:", mse(real, reconstructed))
```

That gave the following for CNN Diffusion
```Text
MSE: 0.0008771765
```
And the next for the U-Net Model
```Text

```

To evaluate the ability of the neural network to preserve the spatial structure of collisions, I implemented the SSIM metric. The function individually processes each pair of matrices (real and reconstructed), adjusts its format and restricts its values to the range of 0 to 1, finally calculating a global average that robustly quantifies the topological fidelity of the model.

```Python
def compute_ssim(x, y):
    scores = []
    for i in range(len(x)):
        xi = x[i].transpose(1,2,0)
        yi = y[i].transpose(1,2,0)

        xi = np.clip(xi, 0, 1)
        yi = np.clip(yi, 0, 1)

        scores.append(ssim(xi, yi, channel_axis=2, data_range=1.0))
    
    return np.mean(scores)

print("SSIM:", compute_ssim(real, reconstructed))
```

That gave the following result for CNN Diffusion
```Text
SSIM: 0.37519923
```

And the next for the U-Net Model
```Text

```


To validate the statistical fidelity of the model beyond the spatial topology, I implemented a comparative histogram of the pixel intensities.
```Python
plt.hist(real.flatten(), bins=50, alpha=0.5, label="Real")
plt.hist(generated.flatten(), bins=50, alpha=0.5, label="Generated")
plt.legend()
plt.title("Pixel Intensity Distribution")
plt.show()


```

The following graphic is for CNN Difussion
<img width="578" height="435" alt="image" src="https://github.com/user-attachments/assets/68df4df0-cd18-4408-85d9-db4840de7082" />


And this for U-Net model









