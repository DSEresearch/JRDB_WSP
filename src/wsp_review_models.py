from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_features(delta_t: torch.Tensor, time_scale: float) -> torch.Tensor:
    tau = delta_t.clamp_min(0.0).unsqueeze(-1) / float(time_scale)
    frequencies = tau * torch.tensor(
        [1.0, 2.0, 4.0, 8.0], device=tau.device, dtype=tau.dtype
    )
    return torch.cat(
        (tau, torch.log1p(tau), torch.sin(math.pi * frequencies),
         torch.cos(math.pi * frequencies)),
        dim=-1,
    )


class SequenceDynamicsBase(nn.Module):
    """Common interface for teacher-forced sequence dynamics models."""

    model_name = "base"

    def forward_sequence(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim == 2:
            return self.forward_sequence(
                state.unsqueeze(1), action.unsqueeze(1), delta_t.unsqueeze(1)
            )[:, 0]
        return self.forward_sequence(state, action, delta_t)

    def config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}


class TimeMLPSequence(SequenceDynamicsBase):
    model_name = "time_mlp"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 d_model: int, depth: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.depth = depth
        layers: list[nn.Module] = [nn.Linear(state_dim + action_dim + 10, d_model)]
        for _ in range(depth - 1):
            layers.extend((nn.GELU(), nn.Linear(d_model, d_model)))
        layers.extend((nn.GELU(), nn.Linear(d_model, state_dim)))
        self.network = nn.Sequential(*layers)

    def forward_sequence(self, state, action, delta_t):
        return self.network(
            torch.cat((state, action, time_features(delta_t, self.time_scale)), dim=-1)
        )

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model, "depth": self.depth}


class SineTimeSequence(SequenceDynamicsBase):
    model_name = "sine_mlp"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 d_model: int, depth: int = 3, first_omega: float = 10.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.depth = depth
        self.first_omega = float(first_omega)
        self.layers = nn.ModuleList()
        input_dim = state_dim + action_dim + 1
        for index in range(depth):
            layer = nn.Linear(input_dim if index == 0 else d_model, d_model)
            bound = 1.0 / max(1, layer.in_features)
            nn.init.uniform_(layer.weight, -bound, bound)
            self.layers.append(layer)
        self.readout = nn.Linear(d_model, state_dim)

    def forward_sequence(self, state, action, delta_t):
        tau = delta_t.unsqueeze(-1) / self.time_scale
        hidden = torch.cat((state, action, tau), dim=-1)
        for index, layer in enumerate(self.layers):
            hidden = torch.sin((self.first_omega if index == 0 else 1.0) * layer(hidden))
        return self.readout(hidden)

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model, "depth": self.depth,
                "first_omega": self.first_omega}


class WFBPointwiseSequence(SequenceDynamicsBase):
    """Pointwise WFB response under the same state-transition target as WSP."""

    model_name = "wfb_pointwise"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 d_model: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.amplitude_raw = nn.Parameter(torch.full((d_model,), -1.0))
        self.k_raw = nn.Parameter(torch.linspace(-1.0, 1.0, d_model))
        self.omega_raw = nn.Parameter(torch.linspace(-1.0, 1.0, d_model))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, d_model))
        self.readout = nn.Linear(d_model, state_dim)

    def forward_sequence(self, state, action, delta_t):
        content = self.input_projection(torch.cat((state, action), dim=-1))
        tau = delta_t.unsqueeze(-1) / self.time_scale
        amplitude = F.softplus(self.amplitude_raw)
        k = F.softplus(self.k_raw)
        omega = 2.0 * math.pi * torch.sigmoid(self.omega_raw)
        wave = amplitude * torch.cos(k * content - omega * tau - self.theta)
        return self.readout(wave)

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model}


class TimeGRUSequence(SequenceDynamicsBase):
    model_name = "time_gru"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 d_model: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.input_projection = nn.Linear(state_dim + action_dim + 10, d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.readout = nn.Linear(d_model, state_dim)

    def forward_sequence(self, state, action, delta_t):
        batch, steps, _ = state.shape
        hidden = state.new_zeros(batch, self.d_model)
        outputs = []
        inputs = self.input_projection(
            torch.cat((state, action, time_features(delta_t, self.time_scale)), dim=-1)
        )
        for index in range(steps):
            hidden = self.cell(inputs[:, index], hidden)
            outputs.append(self.readout(hidden))
        return torch.stack(outputs, dim=1)

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model}


class ComplexSSMSequence(SequenceDynamicsBase):
    """Global diagonal complex continuous-time SSM without WSP conditioning."""

    model_name = "complex_ssm"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 target_mean: np.ndarray, target_std: np.ndarray, d_model: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.input_projection = nn.Linear(state_dim + action_dim, 2 * d_model)
        self.omega_raw = nn.Parameter(torch.linspace(-1.5, 1.5, d_model))
        self.gamma_raw = nn.Parameter(torch.full((d_model,), -3.0))
        self.readout = nn.Linear(2 * d_model, state_dim, bias=False)
        zero_target = -np.asarray(target_mean, dtype=np.float32) / np.maximum(
            np.asarray(target_std, dtype=np.float32), 1e-8
        )
        self.register_buffer("zero_target", torch.as_tensor(zero_target))

    def forward_sequence(self, state, action, delta_t):
        batch = state.shape[0]
        carry = state.new_zeros(batch, self.d_model, 2)
        omega = 2.0 * math.pi * torch.sigmoid(self.omega_raw)
        gamma = 2.0 * torch.sigmoid(self.gamma_raw)
        injections = self.input_projection(torch.cat((state, action), dim=-1))
        injections = injections.reshape(batch, state.shape[1], self.d_model, 2)
        outputs = []
        for index in range(state.shape[1]):
            posterior = carry + injections[:, index]
            tau = delta_t[:, index].unsqueeze(-1) / self.time_scale
            phase = -omega * tau
            decay = torch.exp(-gamma * tau)
            u, v = posterior.unbind(dim=-1)
            propagated = torch.stack((
                decay * (u * torch.cos(phase) - v * torch.sin(phase)),
                decay * (u * torch.sin(phase) + v * torch.cos(phase)),
            ), dim=-1)
            outputs.append(
                self.readout((propagated - posterior).flatten(start_dim=1))
                + self.zero_target
            )
            carry = propagated
        return torch.stack(outputs, dim=1)

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model}


class CausalTransformerSequence(SequenceDynamicsBase):
    model_name = "transformer_matched"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 d_model: int, num_layers: int = 2, nhead: int = 4,
                 ffn_multiplier: int = 2, max_length: int = 512):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.ffn_multiplier = ffn_multiplier
        self.max_length = max_length
        self.input_projection = nn.Linear(state_dim + action_dim + 10, d_model)
        self.position = nn.Parameter(torch.zeros(max_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * ffn_multiplier, 0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, nn.LayerNorm(d_model))
        self.readout = nn.Linear(d_model, state_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward_sequence(self, state, action, delta_t):
        steps = state.shape[1]
        if steps > self.max_length:
            raise ValueError(f"Sequence length {steps} exceeds {self.max_length}")
        hidden = self.input_projection(
            torch.cat((state, action, time_features(delta_t, self.time_scale)), dim=-1)
        ) + self.position[:steps]
        mask = torch.full((steps, steps), float("-inf"), device=state.device)
        mask = torch.triu(mask, diagonal=1)
        return self.readout(self.encoder(hidden, mask=mask, is_causal=True))

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model, "num_layers": self.num_layers,
                "nhead": self.nhead, "ffn_multiplier": self.ffn_multiplier,
                "max_length": self.max_length}


@dataclass(frozen=True)
class WSPAblation:
    carry: bool = True
    frequency: bool = True
    damping: bool = True
    position_phase: bool = True
    conditioner: bool = True
    zero_time: bool = True
    complex_pair: bool = True


class PersistentWaveLayer(nn.Module):
    def __init__(self, d_model: int, condition_rank: int, omega_max: float,
                 ablation: WSPAblation):
        super().__init__()
        self.d_model = d_model
        self.omega_max = float(omega_max)
        self.ablation = ablation
        self.norm = nn.LayerNorm(d_model)
        self.drive = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.pair_readout = nn.Linear(2 * d_model, d_model, bias=False)
        self.condition_down = nn.Linear(d_model, condition_rank, bias=False)
        self.condition_up = nn.Linear(condition_rank, 5 * d_model, bias=False)
        self.amplitude_raw = nn.Parameter(torch.full((d_model,), -1.0))
        self.k_raw = nn.Parameter(torch.linspace(-1.0, 1.0, d_model))
        self.omega_raw = nn.Parameter(torch.linspace(-1.5, 1.5, d_model))
        self.gamma_raw = nn.Parameter(torch.full((d_model,), -3.0))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, d_model))
        nn.init.zeros_(self.condition_up.weight)

    def parameters_for(self, content: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.ablation.conditioner:
            shifts = self.condition_up(self.condition_down(content)).chunk(5, dim=-1)
        else:
            zero = torch.zeros_like(content)
            shifts = (zero, zero, zero, zero, zero)
        amplitude = F.softplus(self.amplitude_raw + shifts[0])
        k = F.softplus(self.k_raw + shifts[1])
        omega = self.omega_max * torch.sigmoid(self.omega_raw + shifts[2])
        gamma = 2.0 * torch.sigmoid(self.gamma_raw + shifts[3])
        theta = self.theta + math.pi * torch.tanh(shifts[4])
        if not self.ablation.position_phase:
            k = torch.zeros_like(k)
        if not self.ablation.frequency:
            omega = torch.zeros_like(omega)
        if not self.ablation.damping:
            gamma = torch.zeros_like(gamma)
        return amplitude, k, omega, gamma, theta

    def step(self, content: torch.Tensor, tau: torch.Tensor,
             carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(content)
        amplitude, k, omega, gamma, theta = self.parameters_for(normalized)
        drive = amplitude * self.drive(normalized)
        gate = torch.sigmoid(self.gate(normalized))
        injection_phase = k * normalized + theta
        injection_u = gate * drive * torch.cos(injection_phase)
        injection_v = gate * drive * torch.sin(injection_phase)
        if not self.ablation.complex_pair:
            injection_v = torch.zeros_like(injection_v)
        posterior = carry + torch.stack((injection_u, injection_v), dim=-1)
        # For state dynamics there is no independent token-position increment.
        # k shapes the state-conditioned injection phase; elapsed physical time
        # alone controls propagation, preserving exact identity at delta_t=0.
        phase = -omega * tau.unsqueeze(-1)
        decay = torch.exp(-gamma * tau.unsqueeze(-1))
        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)
        u, v = posterior.unbind(dim=-1)
        propagated_u = decay * (u * cos_phase - v * sin_phase)
        propagated_v = decay * (u * sin_phase + v * cos_phase)
        if not self.ablation.complex_pair:
            propagated_v = torch.zeros_like(propagated_v)
        propagated = torch.stack((propagated_u, propagated_v), dim=-1)
        difference = propagated - posterior
        update = self.pair_readout(difference.flatten(start_dim=-2))
        return content + update, propagated


class PersistentWSPSequence(SequenceDynamicsBase):
    model_name = "wsp_carry"

    def __init__(self, state_dim: int, action_dim: int, time_scale: float,
                 target_mean: np.ndarray, target_std: np.ndarray,
                 d_model: int, num_layers: int = 2, condition_rank: int = 16,
                 omega_max: float = 2.0 * math.pi,
                 ablation: WSPAblation | None = None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_scale = float(time_scale)
        self.d_model = d_model
        self.num_layers = num_layers
        self.condition_rank = condition_rank
        self.omega_max = float(omega_max)
        self.ablation = ablation or WSPAblation()
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.layers = nn.ModuleList(
            PersistentWaveLayer(d_model, condition_rank, omega_max, self.ablation)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, state_dim, bias=False)
        zero_target = -np.asarray(target_mean, dtype=np.float32) / np.maximum(
            np.asarray(target_std, dtype=np.float32), 1e-8
        )
        self.register_buffer("zero_target", torch.as_tensor(zero_target))

    def initial_carry(self, batch: int, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            batch, self.num_layers, self.d_model, 2, device=device, dtype=dtype
        )

    def forward_step(self, state: torch.Tensor, action: torch.Tensor,
                     delta_t: torch.Tensor, carry: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        if carry is None or not self.ablation.carry:
            carry = self.initial_carry(batch, state.device, state.dtype)
        initial = self.input_projection(torch.cat((state, action), dim=-1))
        content = initial
        tau = delta_t.clamp_min(0.0) / self.time_scale
        next_carries = []
        for index, layer in enumerate(self.layers):
            content, next_carry = layer.step(content, tau, carry[:, index])
            next_carries.append(next_carry)
        if self.ablation.zero_time:
            representation = self.final_norm(content) - self.final_norm(initial)
            prediction = self.readout(representation) + self.zero_target
        else:
            prediction = self.readout(self.final_norm(content))
        return prediction, torch.stack(next_carries, dim=1)

    def forward_sequence(self, state, action, delta_t):
        carry = None
        outputs = []
        for index in range(state.shape[1]):
            prediction, carry = self.forward_step(
                state[:, index], action[:, index], delta_t[:, index], carry
            )
            outputs.append(prediction)
        return torch.stack(outputs, dim=1)

    def config(self):
        return {**super().config(), "state_dim": self.state_dim,
                "action_dim": self.action_dim, "time_scale": self.time_scale,
                "d_model": self.d_model, "num_layers": self.num_layers,
                "condition_rank": self.condition_rank,
                "omega_max": self.omega_max,
                "ablation": asdict(self.ablation)}


MODEL_ALIASES = {
    "wsp_carry": WSPAblation(),
    "wsp_reset": WSPAblation(carry=False),
    "wsp_no_frequency": WSPAblation(frequency=False),
    "wsp_no_damping": WSPAblation(damping=False),
    "wsp_no_position": WSPAblation(position_phase=False),
    "wsp_no_conditioner": WSPAblation(conditioner=False),
    "wsp_no_zero_time": WSPAblation(zero_time=False),
    "wsp_real_only": WSPAblation(complex_pair=False),
}


def build_review_model(model_name: str, *, state_dim: int, action_dim: int,
                       time_scale: float, target_mean: np.ndarray,
                       target_std: np.ndarray, d_model: int = 128,
                       num_layers: int = 2, condition_rank: int = 16,
                       max_length: int = 512) -> SequenceDynamicsBase:
    common = dict(state_dim=state_dim, action_dim=action_dim,
                  time_scale=time_scale, d_model=d_model)
    if model_name == "time_mlp":
        return TimeMLPSequence(**common, depth=max(2, num_layers))
    if model_name == "sine_mlp":
        return SineTimeSequence(**common, depth=max(2, num_layers))
    if model_name == "wfb_pointwise":
        return WFBPointwiseSequence(**common)
    if model_name == "time_gru":
        return TimeGRUSequence(**common)
    if model_name == "complex_ssm":
        return ComplexSSMSequence(
            **common, target_mean=target_mean, target_std=target_std
        )
    if model_name == "transformer_matched":
        nhead = next(head for head in (8, 4, 2, 1) if d_model % head == 0)
        return CausalTransformerSequence(
            **common, num_layers=num_layers, nhead=nhead, max_length=max_length
        )
    if model_name in MODEL_ALIASES:
        model = PersistentWSPSequence(
            **common, target_mean=target_mean, target_std=target_std,
            num_layers=num_layers, condition_rank=condition_rank,
            ablation=MODEL_ALIASES[model_name],
        )
        model.model_name = model_name
        return model
    raise ValueError(f"Unknown review model: {model_name}")


def load_review_model(checkpoint: dict[str, Any]) -> SequenceDynamicsBase:
    config = dict(checkpoint["model_config"])
    model_name = config.pop("model_name")
    config.pop("ablation", None)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    model = build_review_model(
        model_name,
        target_mean=target_mean,
        target_std=target_std,
        **{key: value for key, value in config.items()
           if key in {"state_dim", "action_dim", "time_scale", "d_model",
                      "num_layers", "condition_rank", "max_length"}},
    )
    model.load_state_dict(checkpoint["model_state"])
    return model
