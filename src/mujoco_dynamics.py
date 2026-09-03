from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


def load_mujoco_normalization(env_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(env_dir) / "normalization.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[str, np.ndarray] = {}
    for field in ("state", "action", "delta_t", "state_delta"):
        for statistic in ("mean", "std"):
            result[f"{field}_{statistic}"] = np.asarray(
                payload[field][statistic],
                dtype=np.float32,
            )
    return result


def load_mujoco_metadata(env_dir: str | Path) -> dict:
    with (Path(env_dir) / "dataset_summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MujocoTransitionDataset(IterableDataset):
    """Stream compressed MuJoCo shards without loading a complete split."""

    def __init__(
        self,
        env_dir: str | Path,
        split: str,
        normalization: dict[str, np.ndarray],
        *,
        shuffle: bool,
        seed: int,
        behavior_only: bool = False,
        max_samples: int = 0,
    ):
        super().__init__()
        self.env_dir = Path(env_dir)
        self.split = split
        self.normalization = normalization
        self.shuffle = shuffle
        self.seed = seed
        self.behavior_only = behavior_only
        self.max_samples = max_samples
        self.epoch = 0
        self.shards = sorted((self.env_dir / split).glob("part-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No NPZ shards found in {self.env_dir / split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_shards(self) -> list[Path]:
        worker = get_worker_info()
        if worker is None:
            return self.shards
        return self.shards[worker.id :: worker.num_workers]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        shards = self._worker_shards().copy()
        if self.shuffle:
            rng.shuffle(shards)
        emitted = 0
        state_mean = self.normalization["state_mean"]
        state_std = self.normalization["state_std"]
        action_mean = self.normalization["action_mean"]
        action_std = self.normalization["action_std"]
        target_mean = self.normalization["state_delta_mean"]
        target_std = self.normalization["state_delta_std"]

        for path in shards:
            with np.load(path) as payload:
                count = payload["state"].shape[0]
                indices = np.arange(count)
                if self.behavior_only:
                    indices = indices[payload["is_behavior_transition"]]
                if self.shuffle:
                    rng.shuffle(indices)
                for index in indices:
                    state = payload["state"][index].astype(np.float32, copy=False)
                    action = payload["action"][index].astype(np.float32, copy=False)
                    target = payload["state_delta"][index].astype(np.float32, copy=False)
                    yield {
                        "state": torch.from_numpy((state - state_mean) / state_std),
                        "action": torch.from_numpy((action - action_mean) / action_std),
                        "delta_t": torch.tensor(payload["delta_t"][index], dtype=torch.float32),
                        "target": torch.from_numpy((target - target_mean) / target_std),
                        "target_raw": torch.from_numpy(target.copy()),
                        "next_state_raw": torch.from_numpy(
                            payload["next_state"][index].astype(np.float32, copy=True)
                        ),
                        "group_id": torch.tensor(payload["group_id"][index], dtype=torch.int64),
                        "episode_id": torch.tensor(payload["episode_id"][index], dtype=torch.int64),
                        "decision_index": torch.tensor(
                            payload["decision_index"][index],
                            dtype=torch.int64,
                        ),
                    }
                    emitted += 1
                    if self.max_samples and emitted >= self.max_samples:
                        return


class MujocoTransitionBatchDataset(IterableDataset):
    """Vectorized shard streamer that yields already assembled training batches."""

    def __init__(
        self,
        env_dir: str | Path,
        split: str,
        normalization: dict[str, np.ndarray],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_last: bool,
        max_samples: int = 0,
    ):
        super().__init__()
        self.env_dir = Path(env_dir)
        self.split = split
        self.normalization = normalization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.max_samples = max_samples
        self.epoch = 0
        self.shards = sorted((self.env_dir / split).glob("part-*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No NPZ shards found in {self.env_dir / split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_context(self) -> tuple[list[Path], int]:
        worker = get_worker_info()
        if worker is None:
            return self.shards, self.max_samples
        maximum = (
            math.ceil(self.max_samples / worker.num_workers)
            if self.max_samples
            else 0
        )
        return self.shards[worker.id :: worker.num_workers], maximum

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        worker_shards, worker_maximum = self._worker_context()
        shards = worker_shards.copy()
        if self.shuffle:
            rng.shuffle(shards)
        emitted = 0
        norm = self.normalization

        for path in shards:
            with np.load(path) as payload:
                indices = np.arange(payload["state"].shape[0])
                if self.shuffle:
                    rng.shuffle(indices)
                if worker_maximum:
                    remaining = worker_maximum - emitted
                    if remaining <= 0:
                        return
                    indices = indices[:remaining]
                for start in range(0, indices.size, self.batch_size):
                    selected = indices[start : start + self.batch_size]
                    if selected.size < self.batch_size and self.drop_last:
                        continue
                    state_raw = payload["state"][selected].astype(np.float32, copy=False)
                    action_raw = payload["action"][selected].astype(np.float32, copy=False)
                    target_raw = payload["state_delta"][selected].astype(np.float32, copy=False)
                    yield {
                        "state": torch.from_numpy(
                            ((state_raw - norm["state_mean"]) / norm["state_std"]).copy()
                        ),
                        "action": torch.from_numpy(
                            ((action_raw - norm["action_mean"]) / norm["action_std"]).copy()
                        ),
                        "delta_t": torch.from_numpy(
                            payload["delta_t"][selected].astype(np.float32, copy=True)
                        ),
                        "target": torch.from_numpy(
                            (
                                (target_raw - norm["state_delta_mean"])
                                / norm["state_delta_std"]
                            ).copy()
                        ),
                        "target_raw": torch.from_numpy(target_raw.copy()),
                        "next_state_raw": torch.from_numpy(
                            payload["next_state"][selected].astype(np.float32, copy=True)
                        ),
                        "group_id": torch.from_numpy(
                            payload["group_id"][selected].astype(np.int64, copy=True)
                        ),
                        "episode_id": torch.from_numpy(
                            payload["episode_id"][selected].astype(np.int64, copy=True)
                        ),
                        "decision_index": torch.from_numpy(
                            payload["decision_index"][selected].astype(np.int64, copy=True)
                        ),
                    }
                    emitted += selected.size


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return values + torch.log(-torch.expm1(-values))


def _logit(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(values) - torch.log1p(-values)


class TimeConditionedTransformerDynamics(nn.Module):
    """State, action, and physical-time tokens followed by a linear state readout."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.state_projection = nn.Linear(state_dim, d_model)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.time_projection = nn.Linear(4, d_model)
        self.token_type = nn.Parameter(torch.zeros(3, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.readout = nn.Linear(3 * d_model, state_dim)
        nn.init.normal_(self.token_type, std=0.02)

    @staticmethod
    def time_features(delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.unsqueeze(-1)
        return torch.cat(
            (
                dt,
                torch.log1p(dt),
                torch.sin(10.0 * dt),
                torch.cos(10.0 * dt),
            ),
            dim=-1,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            (
                self.state_projection(state),
                self.action_projection(action),
                self.time_projection(self.time_features(delta_t)),
            ),
            dim=1,
        )
        hidden = self.encoder(tokens + self.token_type.unsqueeze(0))
        return self.readout(hidden.flatten(start_dim=1))


class SimNorm(nn.Module):
    """TD-MPC2-style simplicial normalization for a flat latent vector."""

    def __init__(self, latent_dim: int, group_dim: int = 8):
        super().__init__()
        if latent_dim % group_dim:
            raise ValueError("latent_dim must be divisible by group_dim")
        self.group_dim = int(group_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = (*values.shape[:-1], values.shape[-1] // self.group_dim, self.group_dim)
        return torch.softmax(values.reshape(shape), dim=-1).flatten(start_dim=-2)


def physical_time_features(delta_t: torch.Tensor, time_scale: float) -> torch.Tensor:
    tau = (delta_t.clamp_min(0.0) / time_scale).unsqueeze(-1)
    return torch.cat(
        (
            tau,
            torch.log1p(tau),
            torch.sin(math.pi * tau),
            torch.cos(math.pi * tau),
        ),
        dim=-1,
    )


class TDMPCTwoDynamicsProbe(nn.Module):
    """Dynamics-only TD-MPC2 adaptation with a common raw-state probe.

    This preserves the SimNorm latent transition used by TD-MPC2 while adding
    explicit elapsed time and a supervised state-displacement probe. It is not
    the native TD-MPC2 control agent, which also learns reward, value, policy,
    and planning components.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        d_model: int = 128,
        hidden_dim: int = 512,
        simnorm_dim: int = 8,
    ):
        super().__init__()
        self.time_scale = float(time_scale)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, d_model),
            SimNorm(d_model, simnorm_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(d_model + action_dim + 4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, d_model),
            SimNorm(d_model, simnorm_dim),
        )
        self.probe = nn.Linear(2 * d_model, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encoder(state)
        time = physical_time_features(delta_t, self.time_scale)
        next_latent = self.dynamics(torch.cat((latent, action, time), dim=-1))
        return self.probe(torch.cat((latent, next_latent), dim=-1))


class StraightThroughScalarQuantizer(nn.Module):
    """Finite scalar quantization with a straight-through gradient."""

    def __init__(self, levels: int = 8):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2")
        self.levels = int(levels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(values)
        half = (self.levels - 1) / 2.0
        quantized = torch.round(bounded * half) / half
        return bounded + (quantized - bounded).detach()


class DCWMDynamicsProbe(nn.Module):
    """Dynamics-only discrete-codebook world model with a common state probe.

    The finite scalar quantizer follows the discrete-latent design used by
    DCWM. The reward/value heads and decision-time DC-MPC planner are excluded
    so that this class can be evaluated on exactly the same transition target
    as WSP.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        d_model: int = 128,
        hidden_dim: int = 512,
        levels: int = 8,
    ):
        super().__init__()
        self.time_scale = float(time_scale)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.quantizer = StraightThroughScalarQuantizer(levels)
        self.dynamics = nn.Sequential(
            nn.Linear(d_model + action_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.probe = nn.Linear(2 * d_model, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.quantizer(self.encoder(state))
        time = physical_time_features(delta_t, self.time_scale)
        next_latent = self.quantizer(
            self.dynamics(torch.cat((latent, action, time), dim=-1))
        )
        return self.probe(torch.cat((latent, next_latent), dim=-1))


class ContinuousTimeTransformerLayer(nn.Module):
    """ODE-propagated tokens followed by Transformer attention."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ode_norm = nn.LayerNorm(d_model)
        self.vector_field = nn.Sequential(
            nn.Linear(d_model + 2, ffn_dim),
            nn.Tanh(),
            nn.Linear(ffn_dim, d_model),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def field(self, tokens: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_input = torch.stack((time, torch.sin(math.pi * time)), dim=-1)
        time_input = time_input.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        return self.vector_field(torch.cat((self.ode_norm(tokens), time_input), dim=-1))

    def forward(
        self,
        tokens: torch.Tensor,
        start_time: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        # One RK4 step gives a deterministic continuous-time propagation layer.
        step_view = step.view(-1, 1, 1)
        k1 = self.field(tokens, start_time)
        k2 = self.field(tokens + 0.5 * step_view * k1, start_time + 0.5 * step)
        k3 = self.field(tokens + 0.5 * step_view * k2, start_time + 0.5 * step)
        k4 = self.field(tokens + step_view * k3, start_time + step)
        tokens = tokens + step_view * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        normalized = self.attn_norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class ContiFormerDynamicsProbe(nn.Module):
    """ContiFormer-style continuous-time attention for transition prediction.

    This is a dynamics-task adaptation of the published architecture: state and
    action tokens evolve through Neural-ODE RK4 steps and interact through
    multi-head attention. The final linear probe targets normalized state
    displacement under the common benchmark protocol.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_scale = float(time_scale)
        self.state_projection = nn.Linear(state_dim, d_model)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.token_type = nn.Parameter(torch.zeros(2, d_model))
        self.layers = nn.ModuleList(
            ContinuousTimeTransformerLayer(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.probe = nn.Linear(2 * d_model, state_dim)
        nn.init.normal_(self.token_type, std=0.02)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            (self.state_projection(state), self.action_projection(action)),
            dim=1,
        ) + self.token_type.unsqueeze(0)
        normalized_time = delta_t.clamp_min(0.0) / self.time_scale
        step = normalized_time / len(self.layers)
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens, index * step, step)
        return self.probe(self.final_norm(tokens).flatten(start_dim=1))


class WSPDynamicsLayer(nn.Module):
    """Content-dependent wave state propagated directly by elapsed physical time."""

    def __init__(self, d_model: int, minimum_dt: float, maximum_dt: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.wave_readout = nn.Linear(2 * d_model, d_model, bias=False)
        self.amplitude = nn.Parameter(torch.full((d_model,), 0.1))
        k_initial = torch.logspace(math.log10(0.1), math.log10(1.0), d_model)
        omega_low = max(0.5 / maximum_dt, 1e-2)
        omega_high = max(math.pi / minimum_dt, omega_low * 1.01)
        omega_initial = torch.logspace(
            math.log10(omega_low),
            math.log10(omega_high),
            d_model,
        )
        gamma_initial = torch.full((d_model,), 0.02 / maximum_dt)
        self.k_raw = nn.Parameter(_inverse_softplus(k_initial))
        self.omega_raw = nn.Parameter(_inverse_softplus(omega_initial))
        self.gamma_raw = nn.Parameter(_inverse_softplus(gamma_initial))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, d_model))

        with torch.no_grad():
            self.wave_readout.weight.zero_()
            scale = 1.0 / math.sqrt(2.0)
            identity = torch.eye(d_model) * scale
            self.wave_readout.weight[:, :d_model].copy_(identity)
            self.wave_readout.weight[:, d_model:].copy_(identity)

    def wave_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "A": self.amplitude,
            "k": F.softplus(self.k_raw),
            "omega": F.softplus(self.omega_raw),
            "gamma": F.softplus(self.gamma_raw),
            "theta": self.theta,
        }

    def forward(self, hidden: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        content = self.norm(hidden)
        parameters = self.wave_parameters()
        dt = delta_t.clamp_min(0.0).unsqueeze(-1)
        phase = parameters["k"] * content - parameters["omega"] * dt + parameters["theta"]
        envelope = parameters["A"] * torch.exp(-parameters["gamma"] * dt)
        cosine = envelope * content * torch.cos(phase)
        sine = envelope * content * torch.sin(phase)
        return hidden + self.wave_readout(torch.cat((cosine, sine), dim=-1))


class WSPStateDynamics(nn.Module):
    """Wave-only hidden dynamics with a linear state-delta readout."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        minimum_dt: float,
        maximum_dt: float,
        d_model: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.layers = nn.ModuleList(
            WSPDynamicsLayer(d_model, minimum_dt, maximum_dt)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_projection(torch.cat((state, action), dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, delta_t)
        return self.readout(self.final_norm(hidden))

    def wave_parameter_summary(self) -> dict[str, float]:
        grouped: dict[str, list[torch.Tensor]] = {}
        for layer in self.layers:
            for name, values in layer.wave_parameters().items():
                grouped.setdefault(name, []).append(values.detach().flatten())
        return {
            f"{name}_{statistic}": float(
                getattr(torch.cat(values), statistic)().cpu()
            )
            for name, values in grouped.items()
            for statistic in ("mean", "std", "min", "max")
        }


class WSPDynamicsLayerV2(nn.Module):
    """Smooth bounded-frequency wave increment in normalized physical time."""

    def __init__(
        self,
        d_model: int,
        omega_max: float,
        gamma_max: float = 2.0,
    ):
        super().__init__()
        if omega_max <= 0.0 or gamma_max <= 0.0:
            raise ValueError("omega_max and gamma_max must be positive")
        self.omega_max = float(omega_max)
        self.gamma_max = float(gamma_max)
        self.norm = nn.LayerNorm(d_model)
        self.wave_readout = nn.Linear(2 * d_model, d_model, bias=False)

        amplitude_initial = torch.full((d_model,), 0.1)
        k_initial = torch.logspace(math.log10(0.1), math.log10(1.0), d_model)
        omega_initial = torch.logspace(
            math.log10(0.25),
            math.log10(0.8 * omega_max),
            d_model,
        )
        gamma_initial = torch.full((d_model,), 0.05)
        self.amplitude_raw = nn.Parameter(_inverse_softplus(amplitude_initial))
        self.k_raw = nn.Parameter(_inverse_softplus(k_initial))
        self.omega_raw = nn.Parameter(_logit(omega_initial / omega_max))
        self.gamma_raw = nn.Parameter(_logit(gamma_initial / gamma_max))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, d_model))

        with torch.no_grad():
            self.wave_readout.weight.zero_()
            scale = 1.0 / math.sqrt(2.0)
            identity = torch.eye(d_model) * scale
            self.wave_readout.weight[:, :d_model].copy_(identity)
            self.wave_readout.weight[:, d_model:].copy_(identity)

    def wave_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "A": F.softplus(self.amplitude_raw),
            "k": F.softplus(self.k_raw),
            "omega": self.omega_max * torch.sigmoid(self.omega_raw),
            "gamma": self.gamma_max * torch.sigmoid(self.gamma_raw),
            "theta": self.theta,
        }

    def forward(self, hidden: torch.Tensor, normalized_time: torch.Tensor) -> torch.Tensor:
        content = self.norm(hidden)
        parameters = self.wave_parameters()
        tau = normalized_time.clamp_min(0.0).unsqueeze(-1)
        phase_zero = parameters["k"] * content + parameters["theta"]
        phase_time = phase_zero - parameters["omega"] * tau
        decay = torch.exp(-parameters["gamma"] * tau)

        # The propagated increment is exactly zero when elapsed time is zero.
        cosine_delta = decay * torch.cos(phase_time) - torch.cos(phase_zero)
        sine_delta = decay * torch.sin(phase_time) - torch.sin(phase_zero)
        amplitude = parameters["A"] * content
        wave_increment = torch.cat(
            (amplitude * cosine_delta, amplitude * sine_delta),
            dim=-1,
        )
        return hidden + self.wave_readout(wave_increment)


class WSPStateDynamicsV2(nn.Module):
    """Physically scaled WSP with bounded frequency and zero-time consistency."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int = 128,
        num_layers: int = 4,
        omega_max: float = 2.0 * math.pi,
    ):
        super().__init__()
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        self.time_scale = float(time_scale)
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.layers = nn.ModuleList(
            WSPDynamicsLayerV2(d_model, omega_max=omega_max)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, state_dim)
        nn.init.zeros_(self.readout.bias)

        mean = torch.as_tensor(target_mean, dtype=torch.float32)
        std = torch.as_tensor(target_std, dtype=torch.float32).clamp_min(1e-8)
        self.register_buffer("target_mean_over_std", mean / std)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        normalized_time = delta_t.clamp_min(0.0) / self.time_scale
        hidden = self.input_projection(torch.cat((state, action), dim=-1))
        layer_time = normalized_time / len(self.layers)
        for layer in self.layers:
            hidden = layer(hidden, layer_time)
        reference_delta_normalized = self.readout(self.final_norm(hidden))
        tau = normalized_time.unsqueeze(-1)

        # In raw units: delta(t) = tau * delta(time_scale). This expression is
        # the equivalent value in the existing normalized-target coordinates.
        return (
            tau * reference_delta_normalized
            + (tau - 1.0) * self.target_mean_over_std
        )

    def wave_parameter_summary(self) -> dict[str, float]:
        grouped: dict[str, list[torch.Tensor]] = {}
        for layer in self.layers:
            for name, values in layer.wave_parameters().items():
                grouped.setdefault(name, []).append(values.detach().flatten())
        return {
            f"{name}_{statistic}": float(
                getattr(torch.cat(values), statistic)().cpu()
            )
            for name, values in grouped.items()
            for statistic in ("mean", "std", "min", "max")
        }


class WSPDynamicsLayerV3(nn.Module):
    """State-conditioned wave increment without an explicit time-scaling shortcut."""

    def __init__(
        self,
        d_model: int,
        omega_max: float,
        condition_rank: int = 16,
        gamma_max: float = 2.0,
    ):
        super().__init__()
        if omega_max <= 0.0 or gamma_max <= 0.0:
            raise ValueError("omega_max and gamma_max must be positive")
        if condition_rank <= 0:
            raise ValueError("condition_rank must be positive")
        self.d_model = int(d_model)
        self.omega_max = float(omega_max)
        self.gamma_max = float(gamma_max)
        self.norm = nn.LayerNorm(d_model)
        self.wave_readout = nn.Linear(2 * d_model, d_model, bias=False)
        self.condition_down = nn.Linear(d_model, condition_rank, bias=False)
        self.condition_up = nn.Linear(condition_rank, 5 * d_model, bias=False)

        amplitude_initial = torch.full((d_model,), 0.25)
        k_initial = torch.logspace(math.log10(0.1), math.log10(1.0), d_model)
        omega_initial = torch.logspace(
            math.log10(0.25),
            math.log10(0.8 * omega_max),
            d_model,
        )
        gamma_initial = torch.full((d_model,), 0.05)
        self.amplitude_raw = nn.Parameter(_inverse_softplus(amplitude_initial))
        self.k_raw = nn.Parameter(_inverse_softplus(k_initial))
        self.omega_raw = nn.Parameter(_logit(omega_initial / omega_max))
        self.gamma_raw = nn.Parameter(_logit(gamma_initial / gamma_max))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, d_model))

        with torch.no_grad():
            self.wave_readout.weight.zero_()
            scale = 1.0 / math.sqrt(2.0)
            identity = torch.eye(d_model) * scale
            self.wave_readout.weight[:, :d_model].copy_(identity)
            self.wave_readout.weight[:, d_model:].copy_(identity)
            # Start from the global wave bank, then learn state-action modulation.
            self.condition_up.weight.zero_()

    def wave_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "A": F.softplus(self.amplitude_raw),
            "k": F.softplus(self.k_raw),
            "omega": self.omega_max * torch.sigmoid(self.omega_raw),
            "gamma": self.gamma_max * torch.sigmoid(self.gamma_raw),
            "theta": self.theta,
        }

    def conditioned_wave_parameters(
        self,
        content: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shifts = self.condition_up(self.condition_down(content))
        shift_A, shift_k, shift_omega, shift_gamma, shift_theta = shifts.chunk(5, dim=-1)
        return {
            "A": F.softplus(self.amplitude_raw + shift_A),
            "k": F.softplus(self.k_raw + shift_k),
            "omega": self.omega_max * torch.sigmoid(self.omega_raw + shift_omega),
            "gamma": self.gamma_max * torch.sigmoid(self.gamma_raw + shift_gamma),
            "theta": self.theta + math.pi * torch.tanh(shift_theta),
        }

    def forward(self, hidden: torch.Tensor, normalized_time: torch.Tensor) -> torch.Tensor:
        content = self.norm(hidden)
        parameters = self.conditioned_wave_parameters(content)
        tau = normalized_time.clamp_min(0.0).unsqueeze(-1)
        phase_zero = parameters["k"] * content + parameters["theta"]
        phase_time = phase_zero - parameters["omega"] * tau
        decay = torch.exp(-parameters["gamma"] * tau)
        cosine_delta = decay * torch.cos(phase_time) - torch.cos(phase_zero)
        sine_delta = decay * torch.sin(phase_time) - torch.sin(phase_zero)
        amplitude = parameters["A"] * content
        wave_increment = torch.cat(
            (amplitude * cosine_delta, amplitude * sine_delta),
            dim=-1,
        )
        return hidden + self.wave_readout(wave_increment)


class WSPStateDynamicsV3(nn.Module):
    """State-conditioned physical-time waves with exact zero-time consistency."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int = 128,
        num_layers: int = 4,
        omega_max: float = 2.0 * math.pi,
        condition_rank: int = 16,
    ):
        super().__init__()
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        self.time_scale = float(time_scale)
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.layers = nn.ModuleList(
            WSPDynamicsLayerV3(
                d_model,
                omega_max=omega_max,
                condition_rank=condition_rank,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, state_dim, bias=False)

        mean = torch.as_tensor(target_mean, dtype=torch.float32)
        std = torch.as_tensor(target_std, dtype=torch.float32).clamp_min(1e-8)
        self.register_buffer("target_mean_over_std", mean / std)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        normalized_time = delta_t.clamp_min(0.0) / self.time_scale
        initial_hidden = self.input_projection(torch.cat((state, action), dim=-1))
        hidden = initial_hidden
        layer_time = normalized_time / len(self.layers)
        for layer in self.layers:
            hidden = layer(hidden, layer_time)

        propagated_change = self.final_norm(hidden) - self.final_norm(initial_hidden)
        # At t=0 the normalized prediction is -mean/std, which decodes to zero
        # displacement in raw state units.
        return self.readout(propagated_change) - self.target_mean_over_std

    def wave_parameter_summary(self) -> dict[str, float]:
        grouped: dict[str, list[torch.Tensor]] = {}
        for layer in self.layers:
            for name, values in layer.wave_parameters().items():
                grouped.setdefault(name, []).append(values.detach().flatten())
        return {
            f"{name}_{statistic}": float(
                getattr(torch.cat(values), statistic)().cpu()
            )
            for name, values in grouped.items()
            for statistic in ("mean", "std", "min", "max")
        }


def build_mujoco_dynamics_model(
    model_name: str,
    state_dim: int,
    action_dim: int,
    minimum_dt: float,
    maximum_dt: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    wsp_omega_max: float = 2.0 * math.pi,
    wsp_condition_rank: int = 16,
    simnorm_dim: int = 8,
    dcwm_levels: int = 8,
) -> nn.Module:
    if model_name == "transformer":
        return TimeConditionedTransformerDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
    if model_name == "contiformer_probe":
        return ContiFormerDynamicsProbe(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=maximum_dt,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
    if model_name == "tdmpc2_probe":
        return TDMPCTwoDynamicsProbe(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=maximum_dt,
            d_model=d_model,
            hidden_dim=ffn_dim,
            simnorm_dim=simnorm_dim,
        )
    if model_name == "dcwm_probe":
        return DCWMDynamicsProbe(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=maximum_dt,
            d_model=d_model,
            hidden_dim=ffn_dim,
            levels=dcwm_levels,
        )
    if model_name == "wsp":
        return WSPStateDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            minimum_dt=minimum_dt,
            maximum_dt=maximum_dt,
            d_model=d_model,
            num_layers=num_layers,
        )
    if model_name == "wsp_v2":
        if target_mean is None or target_std is None:
            raise ValueError("wsp_v2 requires target_mean and target_std")
        return WSPStateDynamicsV2(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=maximum_dt,
            target_mean=target_mean,
            target_std=target_std,
            d_model=d_model,
            num_layers=num_layers,
            omega_max=wsp_omega_max,
        )
    if model_name == "wsp_v3":
        if target_mean is None or target_std is None:
            raise ValueError("wsp_v3 requires target_mean and target_std")
        return WSPStateDynamicsV3(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=maximum_dt,
            target_mean=target_mean,
            target_std=target_std,
            d_model=d_model,
            num_layers=num_layers,
            omega_max=wsp_omega_max,
            condition_rank=wsp_condition_rank,
        )
    raise ValueError(f"Unknown MuJoCo dynamics model: {model_name}")
