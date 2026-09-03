from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mujoco_dynamics import TimeConditionedTransformerDynamics, WSPStateDynamicsV3
from src.wsp_review_models import (
    PersistentWSPSequence,
    SequenceDynamicsBase,
    TimeGRUSequence,
    WSPAblation,
    time_features,
)


def _zero_target(target_mean: np.ndarray, target_std: np.ndarray) -> torch.Tensor:
    mean = np.asarray(target_mean, dtype=np.float32)
    std = np.maximum(np.asarray(target_std, dtype=np.float32), 1e-8)
    return torch.as_tensor(-mean / std)


class ZeroDisplacementSequence(SequenceDynamicsBase):
    """The physically valid persistence baseline: raw state displacement is zero."""

    model_name = "zero_displacement"

    def __init__(self, target_mean: np.ndarray, target_std: np.ndarray, **_: Any):
        super().__init__()
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def forward_sequence(self, state, action, delta_t):
        del action, delta_t
        return self.zero_target.expand(*state.shape[:-1], self.zero_target.numel())


class TrainMeanSequence(SequenceDynamicsBase):
    """Predict the train-set mean displacement (zero in normalized target units)."""

    model_name = "train_mean"

    def __init__(self, state_dim: int, **_: Any):
        super().__init__()
        self.state_dim = int(state_dim)

    def forward_sequence(self, state, action, delta_t):
        del action, delta_t
        return state.new_zeros(*state.shape[:-1], self.state_dim)


class ConstantVelocitySequence(SequenceDynamicsBase):
    """Teacher-forced constant-velocity extrapolation from adjacent observations."""

    model_name = "constant_velocity"

    def __init__(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        **_: Any,
    ):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))

    def forward_sequence(self, state, action, delta_t):
        del action
        raw_state = state * self.state_std + self.state_mean
        raw_prediction = torch.zeros_like(raw_state)
        if state.shape[1] > 1:
            previous_delta = raw_state[:, 1:] - raw_state[:, :-1]
            denominator = delta_t[:, :-1].clamp_min(torch.finfo(delta_t.dtype).eps)
            ratio = (delta_t[:, 1:] / denominator).unsqueeze(-1)
            raw_prediction[:, 1:] = previous_delta * ratio
        return (raw_prediction - self.target_mean) / self.target_std


class ZeroTimeMLPSequence(SequenceDynamicsBase):
    """MLP control with the exact boundary q(x,t)-q(x,0)."""

    model_name = "zero_time_mlp"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int,
        num_layers: int = 2,
        **_: Any,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_scale = float(time_scale)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        layers: list[nn.Module] = [
            nn.Linear(state_dim + action_dim + 10, d_model),
            nn.GELU(),
        ]
        for _ in range(max(1, num_layers) - 1):
            layers.extend((nn.Linear(d_model, d_model), nn.GELU()))
        layers.append(nn.Linear(d_model, state_dim, bias=False))
        self.network = nn.Sequential(*layers)
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def _response(self, state, action, delta_t):
        features = time_features(delta_t, self.time_scale)
        return self.network(torch.cat((state, action, features), dim=-1))

    def forward_sequence(self, state, action, delta_t):
        zero = torch.zeros_like(delta_t)
        return self._response(state, action, delta_t) - self._response(
            state, action, zero
        ) + self.zero_target

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "time_scale": self.time_scale,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }


class ZeroTimeRidgeSequence(SequenceDynamicsBase):
    """Fixed ridge probe on content-by-zeroed-time interaction features."""

    model_name = "zero_time_ridge"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        weight: np.ndarray | torch.Tensor | None = None,
        alpha: float = math.nan,
        **_: Any,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_scale = float(time_scale)
        self.alpha = float(alpha)
        feature_dim = (state_dim + action_dim + 1) * 10
        if weight is None:
            weight = np.zeros((state_dim, feature_dim), dtype=np.float32)
        tensor = torch.as_tensor(weight, dtype=torch.float32)
        if tensor.shape != (state_dim, feature_dim):
            raise ValueError(
                f"Ridge weight must have shape {(state_dim, feature_dim)}, got {tuple(tensor.shape)}"
            )
        self.register_buffer("weight", tensor)
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def interaction_features(self, state, action, delta_t):
        content = torch.cat((state, action, torch.ones_like(state[..., :1])), dim=-1)
        zeroed_time = time_features(delta_t, self.time_scale) - time_features(
            torch.zeros_like(delta_t), self.time_scale
        )
        return (content.unsqueeze(-1) * zeroed_time.unsqueeze(-2)).flatten(start_dim=-2)

    def forward_sequence(self, state, action, delta_t):
        features = self.interaction_features(state, action, delta_t)
        return F.linear(features, self.weight) + self.zero_target

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "time_scale": self.time_scale,
            "alpha": self.alpha,
        }


class ZeroTimeTransformerSequence(SequenceDynamicsBase):
    """Causal content encoder with an exactly differenced current-time head.

    The encoder receives only previous intervals; the current interval enters a
    shared local head. Consequently, setting any individual current interval to
    zero gives exactly zero raw displacement without changing its causal context.
    """

    model_name = "zero_time_transformer"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int,
        num_layers: int = 2,
        max_length: int = 512,
        ffn_multiplier: int = 2,
        **_: Any,
    ):
        super().__init__()
        nhead = next(head for head in (8, 4, 2, 1) if d_model % head == 0)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_scale = float(time_scale)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.max_length = int(max_length)
        self.ffn_multiplier = int(ffn_multiplier)
        self.input_projection = nn.Linear(state_dim + action_dim + 10, d_model)
        self.position = nn.Parameter(torch.zeros(max_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, nn.LayerNorm(d_model))
        self.time_head = nn.Sequential(
            nn.Linear(d_model + 10, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim, bias=False),
        )
        nn.init.normal_(self.position, std=0.02)
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def _head(self, context, delta_t):
        return self.time_head(
            torch.cat((context, time_features(delta_t, self.time_scale)), dim=-1)
        )

    def forward_sequence(self, state, action, delta_t):
        steps = state.shape[1]
        if steps > self.max_length:
            raise ValueError(f"Sequence length {steps} exceeds {self.max_length}")
        previous_dt = torch.cat((torch.zeros_like(delta_t[:, :1]), delta_t[:, :-1]), dim=1)
        encoded = self.input_projection(
            torch.cat(
                (state, action, time_features(previous_dt, self.time_scale)), dim=-1
            )
        ) + self.position[:steps]
        mask = torch.full((steps, steps), float("-inf"), device=state.device)
        mask = torch.triu(mask, diagonal=1)
        context = self.encoder(encoded, mask=mask, is_causal=True)
        return self._head(context, delta_t) - self._head(
            context, torch.zeros_like(delta_t)
        ) + self.zero_target

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "time_scale": self.time_scale,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "max_length": self.max_length,
            "ffn_multiplier": self.ffn_multiplier,
        }


class ZeroTimeTokenTransformerSequence(SequenceDynamicsBase):
    """Exact-zero wrapper around the paper's three-token transition Transformer."""

    model_name = "zero_time_token_transformer"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int,
        num_layers: int = 2,
        **_: Any,
    ):
        super().__init__()
        nhead = next(head for head in (8, 4, 2, 1) if d_model % head == 0)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.backbone = TimeConditionedTransformerDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=2 * d_model,
            dropout=0.0,
        )
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def forward_sequence(self, state, action, delta_t):
        batch, steps, _ = state.shape
        flat_state = state.reshape(batch * steps, self.state_dim)
        flat_action = action.reshape(batch * steps, self.action_dim)
        flat_time = delta_t.reshape(batch * steps)
        response = self.backbone(flat_state, flat_action, flat_time)
        response_zero = self.backbone(
            flat_state, flat_action, torch.zeros_like(flat_time)
        )
        prediction = response - response_zero + self.zero_target
        return prediction.reshape(batch, steps, self.state_dim)

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "boundary": "q(state,action,delta_t)-q(state,action,0)",
        }


class DiagonalSSMLayer(nn.Module):
    """Paired-real continuous diagonal SSM common-probe layer.

    This is deliberately described as an S5/LRU-style *adaptation*, not as a
    reproduction of either native architecture. Its shell, layer count, stable
    generator bounds, and inject-propagate-difference order match the WSP probe.
    """

    def __init__(
        self,
        d_model: int,
        condition_rank: int,
        conditioned: bool,
        omega_max: float,
        gamma_max: float,
        frequency: bool,
        damping: bool,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.condition_rank = int(condition_rank)
        self.conditioned = bool(conditioned)
        self.omega_max = float(omega_max)
        self.gamma_max = float(gamma_max)
        self.frequency = bool(frequency)
        self.damping = bool(damping)
        self.norm = nn.LayerNorm(d_model)
        self.input_to_pair = nn.Linear(d_model, 2 * d_model)
        self.pair_readout = nn.Linear(2 * d_model, d_model, bias=False)
        self.omega_raw = nn.Parameter(torch.linspace(-1.5, 1.5, d_model))
        self.gamma_raw = nn.Parameter(torch.full((d_model,), -3.0))
        if conditioned:
            self.condition_down = nn.Linear(d_model, condition_rank, bias=False)
            self.condition_up = nn.Linear(condition_rank, 2 * d_model, bias=False)
            nn.init.zeros_(self.condition_up.weight)
        else:
            self.condition_down = None
            self.condition_up = None

    def generator(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conditioned:
            assert self.condition_down is not None and self.condition_up is not None
            omega_shift, gamma_shift = self.condition_up(
                self.condition_down(normalized)
            ).chunk(2, dim=-1)
        else:
            omega_shift = gamma_shift = torch.zeros_like(normalized)
        omega = self.omega_max * torch.sigmoid(self.omega_raw + omega_shift)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw + gamma_shift)
        if not self.frequency:
            omega = torch.zeros_like(omega)
        if not self.damping:
            gamma = torch.zeros_like(gamma)
        return omega, gamma

    def step(self, content, tau, carry):
        normalized = self.norm(content)
        injection = self.input_to_pair(normalized).reshape(
            normalized.shape[0], self.d_model, 2
        )
        posterior = carry + injection
        omega, gamma = self.generator(normalized)
        phase = -omega * tau.unsqueeze(-1)
        decay = torch.exp(-gamma * tau.unsqueeze(-1))
        cosine, sine = torch.cos(phase), torch.sin(phase)
        real, imaginary = posterior.unbind(dim=-1)
        propagated = torch.stack(
            (
                decay * (real * cosine - imaginary * sine),
                decay * (real * sine + imaginary * cosine),
            ),
            dim=-1,
        )
        increment = propagated - posterior
        update = self.pair_readout(increment.flatten(start_dim=-2))
        return content + update, propagated


class MatchedDiagonalSSMSequence(SequenceDynamicsBase):
    """Layered diagonal complex SSM adaptation matched to WSP's probe shell."""

    model_name = "diag_ssm_global"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int,
        num_layers: int = 2,
        condition_rank: int = 16,
        conditioned: bool = False,
        carry: bool = True,
        frequency: bool = True,
        damping: bool = True,
        omega_max: float = 2.0 * math.pi,
        gamma_max: float = 2.0,
        divide_time_by_layers: bool = False,
        **_: Any,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_scale = float(time_scale)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.condition_rank = int(condition_rank)
        self.conditioned = bool(conditioned)
        self.carry = bool(carry)
        self.frequency = bool(frequency)
        self.damping = bool(damping)
        self.omega_max = float(omega_max)
        self.gamma_max = float(gamma_max)
        self.divide_time_by_layers = bool(divide_time_by_layers)
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.layers = nn.ModuleList(
            DiagonalSSMLayer(
                d_model,
                condition_rank,
                conditioned,
                omega_max,
                gamma_max,
                frequency,
                damping,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, state_dim, bias=False)
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def initial_carry(self, batch, device, dtype):
        return torch.zeros(
            batch, self.num_layers, self.d_model, 2, device=device, dtype=dtype
        )

    def forward_sequence(self, state, action, delta_t):
        batch, steps, _ = state.shape
        carry = self.initial_carry(batch, state.device, state.dtype)
        outputs = []
        for step in range(steps):
            if not self.carry:
                carry = self.initial_carry(batch, state.device, state.dtype)
            initial = self.input_projection(
                torch.cat((state[:, step], action[:, step]), dim=-1)
            )
            content = initial
            tau = delta_t[:, step].clamp_min(0.0) / self.time_scale
            if self.divide_time_by_layers:
                tau = tau / self.num_layers
            next_carries = []
            for layer_index, layer in enumerate(self.layers):
                content, next_carry = layer.step(
                    content, tau, carry[:, layer_index]
                )
                next_carries.append(next_carry)
            representation = self.final_norm(content) - self.final_norm(initial)
            outputs.append(self.readout(representation) + self.zero_target)
            carry = torch.stack(next_carries, dim=1)
        return torch.stack(outputs, dim=1)

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "time_scale": self.time_scale,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "condition_rank": self.condition_rank,
            "conditioned": self.conditioned,
            "carry": self.carry,
            "frequency": self.frequency,
            "damping": self.damping,
            "omega_max": self.omega_max,
            "gamma_max": self.gamma_max,
            "divide_time_by_layers": self.divide_time_by_layers,
        }


class TGNN4IProbeLayer(nn.Module):
    """TGNN4I-inspired observation update plus damped paired-real flow.

    The native method targets irregular temporal graphs. This layer adapts its
    GRU observation assimilation and closed-form decay/rotation block to the
    paper's shared state-transition probe; it is not a native reproduction.
    """

    def __init__(self, d_model: int, omega_max: float, gamma_max: float):
        super().__init__()
        self.d_model = int(d_model)
        self.omega_max = float(omega_max)
        self.gamma_max = float(gamma_max)
        self.norm = nn.LayerNorm(d_model)
        self.input_to_pair = nn.Linear(d_model, 2 * d_model)
        self.observation_update = nn.GRUCell(2 * d_model, 2 * d_model)
        self.pair_readout = nn.Linear(2 * d_model, d_model, bias=False)
        self.omega_raw = nn.Parameter(torch.linspace(-1.5, 1.5, d_model))
        self.gamma_raw = nn.Parameter(torch.full((d_model,), -3.0))

    def step(self, content, tau, carry):
        drive = self.input_to_pair(self.norm(content))
        posterior_flat = self.observation_update(drive, carry.flatten(start_dim=1))
        posterior = posterior_flat.reshape(content.shape[0], self.d_model, 2)
        omega = self.omega_max * torch.sigmoid(self.omega_raw)
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        phase = -omega * tau.unsqueeze(-1)
        decay = torch.exp(-gamma * tau.unsqueeze(-1))
        cosine, sine = torch.cos(phase), torch.sin(phase)
        real, imaginary = posterior.unbind(dim=-1)
        propagated = torch.stack(
            (
                decay * (real * cosine - imaginary * sine),
                decay * (real * sine + imaginary * cosine),
            ),
            dim=-1,
        )
        increment = propagated - posterior
        return content + self.pair_readout(increment.flatten(start_dim=1)), propagated


class TGNN4IProbeSequence(SequenceDynamicsBase):
    """Capacity-matched TGNN4I-style common-probe adaptation."""

    model_name = "tgnn4i_probe"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int,
        num_layers: int = 2,
        omega_max: float = 2.0 * math.pi,
        gamma_max: float = 2.0,
        divide_time_by_layers: bool = False,
        **_: Any,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_scale = float(time_scale)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.omega_max = float(omega_max)
        self.gamma_max = float(gamma_max)
        self.divide_time_by_layers = bool(divide_time_by_layers)
        self.input_projection = nn.Linear(state_dim + action_dim, d_model)
        self.layers = nn.ModuleList(
            TGNN4IProbeLayer(d_model, omega_max, gamma_max)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, state_dim, bias=False)
        self.register_buffer("zero_target", _zero_target(target_mean, target_std))

    def forward_sequence(self, state, action, delta_t):
        batch, steps, _ = state.shape
        carry = state.new_zeros(batch, self.num_layers, self.d_model, 2)
        outputs = []
        for step in range(steps):
            initial = self.input_projection(
                torch.cat((state[:, step], action[:, step]), dim=-1)
            )
            content = initial
            tau = delta_t[:, step].clamp_min(0.0) / self.time_scale
            if self.divide_time_by_layers:
                tau = tau / self.num_layers
            next_carries = []
            for layer_index, layer in enumerate(self.layers):
                content, next_carry = layer.step(content, tau, carry[:, layer_index])
                next_carries.append(next_carry)
            representation = self.final_norm(content) - self.final_norm(initial)
            outputs.append(self.readout(representation) + self.zero_target)
            carry = torch.stack(next_carries, dim=1)
        return torch.stack(outputs, dim=1)

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "time_scale": self.time_scale,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "omega_max": self.omega_max,
            "gamma_max": self.gamma_max,
            "divide_time_by_layers": self.divide_time_by_layers,
            "adaptation_notice": "TGNN4I-style common-probe adaptation; not native TGNN4I",
        }


class TransitionLocalWSPSequence(SequenceDynamicsBase):
    """Sequence-compatible wrapper around the paper's transition-local WSP V3."""

    model_name = "wsp_transition_local"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_scale: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        d_model: int,
        num_layers: int = 4,
        condition_rank: int = 16,
        **_: Any,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_scale = float(time_scale)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.condition_rank = int(condition_rank)
        self.model = WSPStateDynamicsV3(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=time_scale,
            target_mean=target_mean,
            target_std=target_std,
            d_model=d_model,
            num_layers=num_layers,
            condition_rank=condition_rank,
        )

    def forward_sequence(self, state, action, delta_t):
        batch, steps, _ = state.shape
        prediction = self.model(
            state.reshape(batch * steps, -1),
            action.reshape(batch * steps, -1),
            delta_t.reshape(batch * steps),
        )
        return prediction.reshape(batch, steps, self.state_dim)

    def config(self):
        return {
            "model_name": self.model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "time_scale": self.time_scale,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "condition_rank": self.condition_rank,
        }


NEURAL_MODELS = {
    "zero_time_mlp",
    "zero_time_transformer",
    "zero_time_token_transformer",
    "time_gru",
    "tgnn4i_probe",
    "diag_ssm_global",
    "diag_ssm_conditioned",
    "diag_ssm_decay_only",
    "diag_ssm_rotation_only",
    "wsp_reset",
    "wsp_no_conditioner",
    "wsp_decay_only",
    "wsp_rotation_only",
    "wsp_no_zero_time",
    "wsp_cosine_only",
    "wsp_carry",
    "wsp_transition_local",
}

ANALYTIC_MODELS = {
    "zero_displacement",
    "train_mean",
    "constant_velocity",
}

ALL_STRONG_MODELS = NEURAL_MODELS | ANALYTIC_MODELS | {"zero_time_ridge"}

WSP_ABLATIONS = {
    "wsp_reset": WSPAblation(carry=False),
    "wsp_no_conditioner": WSPAblation(conditioner=False),
    "wsp_decay_only": WSPAblation(frequency=False),
    "wsp_rotation_only": WSPAblation(damping=False),
    "wsp_no_zero_time": WSPAblation(zero_time=False),
    "wsp_cosine_only": WSPAblation(complex_pair=False),
    "wsp_carry": WSPAblation(),
}


def build_strong_model(
    model_name: str,
    *,
    state_dim: int,
    action_dim: int,
    time_scale: float,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    d_model: int = 128,
    num_layers: int = 2,
    condition_rank: int = 16,
    max_length: int = 512,
    ridge_weight: np.ndarray | torch.Tensor | None = None,
    ridge_alpha: float = math.nan,
    divide_time_by_layers: bool = False,
) -> SequenceDynamicsBase:
    common: dict[str, Any] = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "time_scale": time_scale,
        "target_mean": target_mean,
        "target_std": target_std,
        "state_mean": state_mean,
        "state_std": state_std,
        "d_model": d_model,
        "num_layers": num_layers,
        "condition_rank": condition_rank,
        "max_length": max_length,
        "divide_time_by_layers": divide_time_by_layers,
    }
    if model_name == "zero_displacement":
        return ZeroDisplacementSequence(**common)
    if model_name == "train_mean":
        return TrainMeanSequence(**common)
    if model_name == "constant_velocity":
        return ConstantVelocitySequence(**common)
    if model_name == "zero_time_mlp":
        return ZeroTimeMLPSequence(**common)
    if model_name == "zero_time_ridge":
        return ZeroTimeRidgeSequence(
            **common, weight=ridge_weight, alpha=ridge_alpha
        )
    if model_name == "zero_time_transformer":
        return ZeroTimeTransformerSequence(**common)
    if model_name == "zero_time_token_transformer":
        return ZeroTimeTokenTransformerSequence(**common)
    if model_name == "time_gru":
        return TimeGRUSequence(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=time_scale,
            d_model=d_model,
        )
    if model_name == "tgnn4i_probe":
        return TGNN4IProbeSequence(**common)
    if model_name.startswith("diag_ssm_"):
        model = MatchedDiagonalSSMSequence(
            **common,
            conditioned=model_name == "diag_ssm_conditioned",
            frequency=model_name != "diag_ssm_decay_only",
            damping=model_name != "diag_ssm_rotation_only",
        )
        model.model_name = model_name
        return model
    if model_name in WSP_ABLATIONS:
        model = PersistentWSPSequence(
            state_dim=state_dim,
            action_dim=action_dim,
            time_scale=time_scale,
            target_mean=target_mean,
            target_std=target_std,
            d_model=d_model,
            num_layers=num_layers,
            condition_rank=condition_rank,
            ablation=WSP_ABLATIONS[model_name],
        )
        model.model_name = model_name
        return model
    if model_name == "wsp_transition_local":
        return TransitionLocalWSPSequence(**common)
    raise ValueError(f"Unknown strong-evaluation model: {model_name}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_metadata(model: nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "class": type(model).__name__,
        "trainable_parameters": trainable_parameter_count(model),
    }
    if hasattr(model, "config"):
        payload["config"] = model.config()
    if isinstance(model, PersistentWSPSequence):
        payload["ablation"] = asdict(model.ablation)
    return payload
