"""Model components: a recurrent-GAN baseline and TimeGAN.

TimeGAN follows Yoon, Jarrett and van der Schaar (NeurIPS 2019), "Time-series
Generative Adversarial Networks". It has four recurrent networks:

    Embedder      x (features)     -> h  (latent)
    Recovery      h (latent)       -> x~ (features)
    Generator     z (noise)        -> e^ (latent)
    Supervisor    h_{1..t} / e^    -> next latent step
    Discriminator latent sequence  -> real / fake logit per time step

The embedder and recovery form an autoencoder, the generator and discriminator
play the usual GAN game in latent space, and the supervisor teaches the latent
sequence its step-to-step dynamics. The losses that combine them live in train.py.

The baseline (`RecurrentGAN`) is a plain GAN with the same recurrent building
blocks operating directly on features, with only the adversarial loss. Comparing
it with TimeGAN isolates what the embedding, reconstruction and supervised losses
add.

This module depends only on PyTorch.
"""
import torch
from torch import nn


class GRUHead(nn.Module):
    """Stacked GRU followed by a linear layer applied at every time step."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation=None,
                 bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True,
                          bidirectional=bidirectional)
        width = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(width, out_dim)
        self.activation = activation

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out)
        return self.activation(out) if self.activation is not None else out


class Embedder(GRUHead):
    """Maps feature sequences (B, T, F) to latent sequences (B, T, H) in (0, 1)."""

    def __init__(self, feat_dim, hidden_dim, num_layers):
        super().__init__(feat_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())


class Recovery(GRUHead):
    """Maps latent sequences (B, T, H) back to features (B, T, F).

    No output activation: features are z-scored and unbounded.
    """

    def __init__(self, feat_dim, hidden_dim, num_layers):
        super().__init__(hidden_dim, hidden_dim, feat_dim, num_layers)


class LatentGenerator(GRUHead):
    """Maps noise sequences (B, T, Z) to latent sequences (B, T, H) in (0, 1)."""

    def __init__(self, noise_dim, hidden_dim, num_layers):
        super().__init__(noise_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())


class Supervisor(GRUHead):
    """Predicts the next latent step from the latent steps so far."""

    def __init__(self, hidden_dim, num_layers):
        super().__init__(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1),
                         nn.Sigmoid())


class SequenceDiscriminator(GRUHead):
    """Bidirectional GRU giving a real/fake logit for every time step (B, T, 1)."""

    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__(in_dim, hidden_dim, 1, num_layers, bidirectional=True)


class TimeGAN(nn.Module):
    """Container for the five TimeGAN networks plus a sampling helper."""

    def __init__(self, feat_dim, hidden_dim=64, num_layers=3, noise_dim=None,
                 feature_adv=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.noise_dim = noise_dim or feat_dim
        self.embedder = Embedder(feat_dim, hidden_dim, num_layers)
        self.recovery = Recovery(feat_dim, hidden_dim, num_layers)
        self.generator = LatentGenerator(self.noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = SequenceDiscriminator(hidden_dim, hidden_dim, num_layers)
        # Optional extension (not in the original paper): a second discriminator that sees
        # decoded *features*, so the recovery network is also penalised for wrong marginals
        self.feature_discriminator = (SequenceDiscriminator(feat_dim, hidden_dim, num_layers)
                                      if feature_adv else None)

    def noise(self, batch, seq_len, device=None):
        return torch.randn(batch, seq_len, self.noise_dim, device=device)

    @torch.no_grad()
    def sample(self, n, seq_len, device=None):
        """Generate n synthetic feature sequences (n, seq_len, feat_dim)."""
        e_hat = self.generator(self.noise(n, seq_len, device))
        h_hat = self.supervisor(e_hat)
        return self.recovery(h_hat)


class RecurrentGAN(nn.Module):
    """Baseline: GRU generator and discriminator working directly on features."""

    def __init__(self, feat_dim, hidden_dim=64, num_layers=3, noise_dim=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.noise_dim = noise_dim or feat_dim
        self.generator = GRUHead(self.noise_dim, hidden_dim, feat_dim, num_layers)
        self.discriminator = SequenceDiscriminator(feat_dim, hidden_dim, num_layers)

    def noise(self, batch, seq_len, device=None):
        return torch.randn(batch, seq_len, self.noise_dim, device=device)

    @torch.no_grad()
    def sample(self, n, seq_len, device=None):
        """Generate n synthetic feature sequences (n, seq_len, feat_dim)."""
        return self.generator(self.noise(n, seq_len, device))


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name, feat_dim, hidden_dim=64, num_layers=3, feature_adv=False):
    """Create 'timegan' or 'baseline' with the given sizes."""
    if name == "timegan":
        return TimeGAN(feat_dim, hidden_dim, num_layers, feature_adv=feature_adv)
    if name == "baseline":
        return RecurrentGAN(feat_dim, hidden_dim, num_layers)
    raise ValueError(f"unknown model '{name}' (expected 'timegan' or 'baseline')")
