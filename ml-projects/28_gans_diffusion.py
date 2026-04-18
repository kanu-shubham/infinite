"""
Project 28: GANs & Diffusion Models
======================================
Generative models learn to CREATE new data — images, music, text.
Two major paradigms:
  1. GANs (2014):        Generator vs Discriminator — adversarial game
  2. Diffusion (2020):   Add noise → learn to denoise → generate

What you'll learn:
- GAN training loop: generator and discriminator alternate
- Mode collapse: the main failure mode of GANs
- DDPM (Denoising Diffusion Probabilistic Model) from scratch
- How Stable Diffusion / DALL-E use the same diffusion principle
- Evaluating generative models (FID, visual inspection)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
#  PART 1: GAN (Generative Adversarial Network)
# ════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Generator G: maps random noise z → fake image.

    Input:  z ~ N(0, I)  — random latent vector (z_dim,)
    Output: fake image   — same shape as real images

    Goal: fool the discriminator into thinking fake images are real.
    """
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, img_dim),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminator D: maps image → probability of being real.

    Input:  image (real or fake)
    Output: scalar in [0, 1] — P(real)

    Goal: correctly identify real images (label=1) and fake (label=0).

    Note: LeakyReLU (not ReLU) — prevents dying neurons in discriminator.
    Note: No BatchNorm in discriminator — causes training instability.
    """
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_gan(n_epochs=20, z_dim=100, batch_size=128, lr=2e-4):
    """
    GAN training loop.

    Each step:
    1. Train Discriminator:
       - Real images → D should output 1
       - Fake images → D should output 0
       loss_D = -[log D(real) + log(1 - D(G(z)))]

    2. Train Generator:
       - Generate fakes → D should output 1 (fool D)
       loss_G = -log D(G(z))

    This is a minimax game:  min_G max_D V(D, G)
    """
    print("── Training GAN on MNIST ──\n")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # normalize to [-1, 1]
    ])
    dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    G = Generator(z_dim=z_dim).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Fixed noise for consistent visualization
    fixed_z = torch.randn(64, z_dim).to(DEVICE)

    g_losses, d_losses = [], []

    for epoch in range(1, n_epochs + 1):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0

        for real_imgs, _ in loader:
            B = real_imgs.size(0)
            real_imgs = real_imgs.view(B, -1).to(DEVICE)

            real_labels = torch.ones(B,  1).to(DEVICE) * 0.9   # label smoothing
            fake_labels = torch.zeros(B, 1).to(DEVICE)

            # ── Train Discriminator ───────────────────────────────────
            z    = torch.randn(B, z_dim).to(DEVICE)
            fake = G(z).detach()          # detach: don't backprop into G yet

            loss_D_real = criterion(D(real_imgs), real_labels)
            loss_D_fake = criterion(D(fake),      fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ───────────────────────────────────────
            z    = torch.randn(B, z_dim).to(DEVICE)
            fake = G(z)
            # Generator wants D to say fake images are real
            loss_G = criterion(D(fake), torch.ones(B, 1).to(DEVICE))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_loss_epoch += loss_G.item()
            d_loss_epoch += loss_D.item()

        g_losses.append(g_loss_epoch / len(loader))
        d_losses.append(d_loss_epoch / len(loader))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | G Loss: {g_losses[-1]:.4f} | D Loss: {d_losses[-1]:.4f}")

    return G, D, g_losses, d_losses, fixed_z


# ════════════════════════════════════════════════════════════════════════════
#  PART 2: Diffusion Model (DDPM from scratch)
# ════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    """Encode the timestep t as a sinusoidal embedding (same idea as positional encoding)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / (half - 1))
        emb = t[:, None].float() * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SimpleUNet1D(nn.Module):
    """
    Tiny U-Net for 1D data (Gaussian blobs) to demonstrate diffusion.
    Real DDPM uses a 2D U-Net for images — same principle, larger network.

    Input:  noisy sample x_t  +  timestep t
    Output: predicted noise ε  (the noise we added at step t)
    """
    def __init__(self, data_dim=2, time_dim=32, hidden=128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


class DDPM:
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Forward process (adding noise — fixed, no learning):
      q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)
      x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε,  ε ~ N(0,I)

    Reverse process (denoising — learned):
      p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)
      The model predicts noise ε_θ(x_t, t) ≈ ε

    Training loss:
      L = E[||ε - ε_θ(sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε, t)||²]
      (predict the noise that was added at step t)
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T

        # Noise schedule (linear)
        betas        = torch.linspace(beta_start, beta_end, T)
        alphas       = 1 - betas
        alpha_bars   = torch.cumprod(alphas, dim=0)   # ᾱ_t = ∏ α_s for s=1..t

        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_ab    = alpha_bars.sqrt()
        self.sqrt_1mab  = (1 - alpha_bars).sqrt()

    def q_sample(self, x0, t):
        """Forward: add noise to x0 at timestep t."""
        device = x0.device
        eps = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_ab[t].to(device).view(-1, 1)
        sqrt_1mab = self.sqrt_1mab[t].to(device).view(-1, 1)
        return sqrt_ab * x0 + sqrt_1mab * eps, eps

    @torch.no_grad()
    def p_sample(self, model, x_t, t_scalar):
        """Reverse: one denoising step."""
        model.eval()
        t = torch.full((x_t.size(0),), t_scalar, dtype=torch.long, device=x_t.device)
        eps_pred = model(x_t, t)

        alpha    = self.alphas[t_scalar].to(x_t.device)
        alpha_bar = self.alpha_bars[t_scalar].to(x_t.device)
        beta     = self.betas[t_scalar].to(x_t.device)

        # Equation 11 in DDPM paper
        coef = (1 - alpha) / (1 - alpha_bar).sqrt()
        mean = (1 / alpha.sqrt()) * (x_t - coef * eps_pred)

        if t_scalar > 0:
            noise = torch.randn_like(x_t)
            return mean + beta.sqrt() * noise
        return mean

    @torch.no_grad()
    def sample(self, model, n_samples, data_dim, device):
        """Full reverse process: noise → data."""
        x = torch.randn(n_samples, data_dim).to(device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x


def train_diffusion():
    print("\n── Training Diffusion Model on 2D Gaussian Mixture ──\n")

    # Create 2D dataset: 8-component Gaussian mixture (like 8 digits' clusters)
    np.random.seed(42)
    centers = np.array([[np.cos(2*np.pi*i/8), np.sin(2*np.pi*i/8)] for i in range(8)]) * 3
    data = []
    for c in centers:
        data.append(np.random.randn(250, 2) * 0.3 + c)
    X = np.vstack(data)
    X = torch.tensor(X, dtype=torch.float32)

    ddpm = DDPM(T=500)
    model = SimpleUNet1D(data_dim=2, time_dim=32, hidden=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(X, batch_size=256, shuffle=True)
    n_epochs = 100
    losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x0 in loader:
            x0 = x0.to(DEVICE)
            t  = torch.randint(0, ddpm.T, (x0.size(0),), device=DEVICE)
            x_t, eps = ddpm.q_sample(x0, t)

            eps_pred = model(x_t, t)
            loss = nn.functional.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d}/{n_epochs} | Loss: {losses[-1]:.4f}")

    return model, ddpm, losses, X.numpy()


def main():
    print("=== GANs & Diffusion Models ===")
    print(f"Device: {DEVICE}\n")

    # ── GAN ──────────────────────────────────────────────────────────────
    print("── GAN Core Concept ──\n")
    print("""
  Generator G:      random noise z → fake image
  Discriminator D:  image → P(real)

  Training (minimax game):
    D wants: D(real) → 1,  D(G(z)) → 0
    G wants: D(G(z)) → 1  (fool D)

  Equilibrium: G generates perfectly realistic images,
               D can't distinguish real from fake (outputs 0.5)

  Problems:
    Mode collapse:   G generates only a few modes (e.g., only "1"s)
    Training instability: G and D must improve at same pace
    """)

    G, D, g_losses, d_losses, fixed_z = train_gan(n_epochs=20)

    # Visualize generated images
    G.eval()
    with torch.no_grad():
        fake_imgs = G(fixed_z).cpu().reshape(-1, 1, 28, 28)
        fake_imgs = (fake_imgs + 1) / 2   # rescale to [0,1]

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_imgs[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("GAN Generated MNIST Digits (Epoch 20)", fontsize=12)
    plt.tight_layout()
    plt.savefig("28_gan_generated.png", dpi=100)
    print("\nSaved GAN generated images to 28_gan_generated.png")

    # GAN training curves
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(d_losses, label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("GAN Training Curves\n(G and D competing — losses oscillate)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("28_gan_losses.png", dpi=100)
    print("Saved GAN loss curves to 28_gan_losses.png")

    # ── Diffusion ─────────────────────────────────────────────────────────
    print("\n── Diffusion Model Core Concept ──\n")
    print("""
  Forward (fixed):  x_0 → x_1 → ... → x_T ≈ N(0,I)
                    Gradually add Gaussian noise over T steps

  Reverse (learned): x_T → x_{T-1} → ... → x_0
                    Train a U-Net to predict and remove noise at each step

  Key insight: instead of learning P(x) directly (hard),
               learn to reverse a simple noising process (easier).

  Stable Diffusion adds:
    - Latent space: compress image to 4x smaller latent first
    - Text conditioning: cross-attention on text embeddings (CLIP)
    - Classifier-free guidance: balance image quality vs text alignment
    """)

    model, ddpm, diff_losses, real_data = train_diffusion()

    # Generate samples
    gen_samples = ddpm.sample(model, n_samples=500, data_dim=2, device=DEVICE)
    gen_samples = gen_samples.cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(real_data[:, 0], real_data[:, 1], s=5, alpha=0.3, c="blue")
    axes[0].set_title("Real Data (8 Gaussian clusters)")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(gen_samples[:, 0], gen_samples[:, 1], s=5, alpha=0.3, c="red")
    axes[1].set_title("Diffusion Generated Samples")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(diff_losses)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE Loss (noise prediction)")
    axes[2].set_title("Diffusion Training Loss")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("DDPM: Learning to Reverse a Diffusion Process", fontsize=13)
    plt.tight_layout()
    plt.savefig("28_diffusion_results.png", dpi=100)
    print("Saved diffusion results to 28_diffusion_results.png")

    # ── Compare GAN vs Diffusion ──────────────────────────────────────────
    print("\n── GAN vs Diffusion: When to Use Which ──\n")
    print(f"{'Aspect':<25} {'GAN':^25} {'Diffusion':^25}")
    print("-" * 75)
    rows = [
        ("Sample speed",   "Fast (one forward pass)",   "Slow (T steps, T=1000)"),
        ("Training",       "Unstable (two models)",     "Stable (single model)"),
        ("Sample quality", "Good, but mode collapse",   "Excellent, diverse"),
        ("Control",        "Hard (conditional GAN)",    "Easy (classifier-free guidance)"),
        ("Uses",           "Face generation, video",    "Stable Diffusion, DALL-E 2+"),
        ("Year dominant",  "2014-2021",                 "2021-present"),
    ]
    for row in rows:
        print(f"{row[0]:<25} {row[1]:^25} {row[2]:^25}")

    print("\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ GAN minimax game — G fools D, D catches G")
    print("  ✓ Label smoothing — prevents D from becoming too confident")
    print("  ✓ DDPM forward process — add noise over T steps")
    print("  ✓ DDPM reverse process — U-Net predicts noise at each step")
    print("  ✓ Sinusoidal time embedding — tell the model which timestep")
    print("  ✓ Diffusion training loss — MSE(predicted noise, actual noise)")


if __name__ == "__main__":
    main()
