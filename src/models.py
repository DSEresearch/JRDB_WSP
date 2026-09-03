from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, dropout: float):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SineLayer(nn.Module):
    """SIREN-style linear layer followed by a sinusoidal activation."""

    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0, first: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.omega_0 = omega_0
        self.first = first
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / self.in_dim if self.first else (6.0 / self.in_dim) ** 0.5 / self.omega_0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SineMLP(nn.Module):
    """Periodic-activation control used to separate WFB from generic sine features."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, omega_0: float = 30.0):
        super().__init__()
        if depth < 2:
            raise ValueError("SineMLP depth must be >= 2")
        layers: list[nn.Module] = [SineLayer(in_dim, hidden_dim, omega_0=omega_0, first=True)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        final = nn.Linear(hidden_dim, out_dim)
        bound = (6.0 / hidden_dim) ** 0.5 / omega_0
        nn.init.uniform_(final.weight, -bound, bound)
        nn.init.uniform_(final.bias, -bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFNTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        feature_dim: int = 6,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
        include_delta_t: bool = False,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.include_delta_t = include_delta_t
        in_dim = obs_len * (feature_dim + int(include_delta_t))
        self.model = MLP(in_dim, hidden_dim, pred_len * 2, depth, dropout)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor | None = None) -> torch.Tensor:
        if self.include_delta_t:
            if delta_t is None:
                raise ValueError("delta_t is required when include_delta_t=True")
            x = torch.cat([x, delta_t.unsqueeze(-1)], dim=-1)
        out = self.model(x.flatten(start_dim=1))
        return out.view(x.shape[0], self.pred_len, 2)


class SineTrajectoryPredictor(nn.Module):
    """SIREN-style feed-forward baseline with optional explicit delta_t input."""

    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        feature_dim: int = 6,
        hidden_dim: int = 256,
        depth: int = 4,
        include_delta_t: bool = True,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.include_delta_t = include_delta_t
        in_dim = obs_len * (feature_dim + int(include_delta_t))
        self.model = SineMLP(in_dim, hidden_dim, pred_len * 2, depth, omega_0=omega_0)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor | None = None) -> torch.Tensor:
        if self.include_delta_t:
            if delta_t is None:
                raise ValueError("delta_t is required when include_delta_t=True")
            x = torch.cat([x, delta_t.unsqueeze(-1)], dim=-1)
        out = self.model(x.flatten(start_dim=1))
        return out.view(x.shape[0], self.pred_len, 2)


class WFBTemporalLayer(nn.Module):
    """Wave-style temporal modulation adapted from the provided WFB reference code."""

    def __init__(self, feature_dim: int, hidden_dim: int, wave_ablation: str = "full"):
        super().__init__()
        valid = {"full", "without_A", "without_k", "without_omega", "without_theta"}
        if wave_ablation not in valid:
            raise ValueError(f"Unknown wave_ablation={wave_ablation}. Valid: {sorted(valid)}")
        self.wave_ablation = wave_ablation
        self.project = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.Ax_raw = nn.Parameter(torch.full((hidden_dim,), -0.1))
        self.At_raw = nn.Parameter(torch.full((hidden_dim,), -0.1))
        self.k_raw = nn.Parameter(torch.full((hidden_dim,), -1.2))
        self.omega_raw = nn.Parameter(torch.full((hidden_dim,), -1.2))
        self.theta_x = nn.Parameter(torch.zeros(hidden_dim))
        self.theta_t = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = torch.tanh(self.norm(self.project(x)))
        t = delta_t.unsqueeze(-1)
        Ax = self.Ax_raw.view(1, 1, -1)
        At = self.At_raw.view(1, 1, -1)
        k = self.k_raw.view(1, 1, -1)
        omega = self.omega_raw.view(1, 1, -1)
        theta_x = self.theta_x.view(1, 1, -1)
        theta_t = self.theta_t.view(1, 1, -1)
        if self.wave_ablation == "without_A":
            Ax = torch.ones_like(Ax)
            At = torch.ones_like(At)
        if self.wave_ablation == "without_k":
            k = torch.ones_like(k)
        if self.wave_ablation == "without_omega":
            omega = torch.zeros_like(omega)
        if self.wave_ablation == "without_theta":
            theta_x = torch.zeros_like(theta_x)
            theta_t = torch.zeros_like(theta_t)
        theta = theta_x + theta_t
        phi = k * z - omega * t - theta
        wave = Ax * At * torch.cos(phi)
        aux = {
            "z": z,
            "t": t,
            "phi": phi,
            "wave": wave,
            "Ax": Ax,
            "At": At,
            "k": k,
            "omega": omega,
            "theta_x": theta_x,
            "theta_t": theta_t,
        }
        return wave, aux


class WFBTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        feature_dim: int = 6,
        wave_dim: int = 128,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
        wave_ablation: str = "full",
        decoder: str = "mlp",
    ):
        super().__init__()
        if decoder not in {"mlp", "linear"}:
            raise ValueError("decoder must be 'mlp' or 'linear'")
        self.pred_len = pred_len
        self.decoder = decoder
        self.wfb = WFBTemporalLayer(feature_dim=feature_dim, hidden_dim=wave_dim, wave_ablation=wave_ablation)
        decoder_depth = depth if decoder == "mlp" else 1
        self.head = MLP(obs_len * wave_dim, hidden_dim, pred_len * 2, decoder_depth, dropout)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wave, aux = self.wfb(x, delta_t)
        out = self.head(wave.flatten(start_dim=1))
        return out.view(x.shape[0], self.pred_len, 2), aux


def laplacian_penalty(aux: dict[str, torch.Tensor]) -> torch.Tensor:
    """Component-wise curvature energy; k is fixed in the temporal correction branch."""
    wave = aux["wave"]
    k = aux["k"].detach()
    return (0.5 * (k.pow(4)) * wave.pow(2)).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(
    model_name: str,
    obs_len: int,
    pred_len: int,
    hidden_dim: int,
    wave_dim: int,
    depth: int,
    dropout: float,
    feature_dim: int = 6,
    wave_ablation: str = "full",
    wfb_decoder: str = "mlp",
    include_delta_t: bool = False,
    sine_omega_0: float = 30.0,
) -> nn.Module:
    if model_name == "ffn":
        return FFNTrajectoryPredictor(
            obs_len,
            pred_len,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            include_delta_t=include_delta_t,
        )
    if model_name == "sine_ffn":
        return SineTrajectoryPredictor(
            obs_len,
            pred_len,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            include_delta_t=include_delta_t,
            omega_0=sine_omega_0,
        )
    if model_name.startswith("wfb"):
        return WFBTrajectoryPredictor(
            obs_len,
            pred_len,
            feature_dim=feature_dim,
            wave_dim=wave_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            wave_ablation=wave_ablation,
            decoder=wfb_decoder,
        )
    raise ValueError(f"Unknown model: {model_name}")
