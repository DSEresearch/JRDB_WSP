from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TRANSFORMER_MODEL_VERSION = "transformer_v1"
WSP_MODEL_VERSION = "wsp_v2_wave_first_gated_phase_attention"


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class TransformerTrajectoryPredictor(nn.Module):
    """Standard Transformer encoder followed by one linear trajectory readout."""

    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = LearnedPositionEmbedding(obs_len, d_model)
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
        self.readout = nn.Linear(obs_len * d_model, pred_len * 2)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.position(self.input_projection(x))
        hidden = self.encoder(hidden)
        return self.readout(hidden.flatten(start_dim=1)).view(x.shape[0], self.pred_len, 2)


class WaveStatePropagation(nn.Module):
    """Real-valued cosine/sine implementation of elapsed-time wave propagation."""

    def __init__(self, d_model: int, scan_mode: str = "sequential"):
        super().__init__()
        if scan_mode not in {"sequential", "parallel"}:
            raise ValueError("scan_mode must be 'sequential' or 'parallel'")
        self.scan_mode = scan_mode
        self.content_norm = nn.LayerNorm(d_model)
        self.content_drive = nn.Linear(d_model, d_model)
        self.content_gate = nn.Linear(d_model, d_model)
        self.wave_readout = nn.Linear(2 * d_model, d_model, bias=False)

        self.amplitude = nn.Parameter(torch.full((d_model,), 0.1))
        k_initial = torch.logspace(math.log10(0.1), math.log10(1.0), d_model)
        omega_initial = torch.logspace(math.log10(0.1), math.log10(2.0), d_model)
        gamma_initial = torch.full((d_model,), 0.02)
        self.k_raw = nn.Parameter(_inverse_softplus(k_initial))
        self.omega_raw = nn.Parameter(_inverse_softplus(omega_initial))
        self.gamma_raw = nn.Parameter(_inverse_softplus(gamma_initial))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, d_model))

        nn.init.eye_(self.content_drive.weight)
        nn.init.zeros_(self.content_drive.bias)
        nn.init.zeros_(self.content_gate.weight)
        nn.init.ones_(self.content_gate.bias)
        with torch.no_grad():
            self.wave_readout.weight.zero_()
            scale = 1.0 / math.sqrt(2.0)
            self.wave_readout.weight[:, :d_model].copy_(torch.eye(d_model) * scale)
            self.wave_readout.weight[:, d_model:].copy_(torch.eye(d_model) * scale)

    def wave_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "A": self.amplitude,
            "k": F.softplus(self.k_raw),
            "omega": F.softplus(self.omega_raw),
            "gamma": F.softplus(self.gamma_raw),
            "theta": self.theta,
        }

    def _sequential_scan(
        self,
        injected_amplitude: torch.Tensor,
        delta_t: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cosine_state = torch.zeros_like(injected_amplitude[:, 0])
        sine_state = torch.zeros_like(injected_amplitude[:, 0])
        theta_cosine = torch.cos(params["theta"])
        theta_sine = torch.sin(params["theta"])
        states = []

        for step in range(injected_amplitude.shape[1]):
            dt = delta_t[:, step]
            phase_increment = params["k"] - params["omega"] * dt
            cos_phase = torch.cos(phase_increment)
            sin_phase = torch.sin(phase_increment)
            decay = torch.exp(-params["gamma"] * dt)

            propagated_cosine = decay * (cosine_state * cos_phase - sine_state * sin_phase)
            propagated_sine = decay * (cosine_state * sin_phase + sine_state * cos_phase)
            cosine_state = propagated_cosine + injected_amplitude[:, step] * theta_cosine
            sine_state = propagated_sine + injected_amplitude[:, step] * theta_sine
            states.append(torch.cat((cosine_state, sine_state), dim=-1))

        return torch.stack(states, dim=1)

    @staticmethod
    def _parallel_scan(
        injected_amplitude: torch.Tensor,
        delta_t: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Closed-form scan equivalent to the recurrent complex rotation."""
        cumulative_time = torch.cumsum(delta_t, dim=1)
        phase_increment = params["k"].view(1, 1, -1) - params["omega"].view(1, 1, -1) * delta_t
        cumulative_phase = torch.cumsum(phase_increment, dim=1)
        log_magnitude = -params["gamma"].view(1, 1, -1) * cumulative_time

        injection_phase = params["theta"].view(1, 1, -1) - cumulative_phase
        inverse_magnitude = torch.exp(-log_magnitude)
        cumulative_cosine = torch.cumsum(
            injected_amplitude * inverse_magnitude * torch.cos(injection_phase),
            dim=1,
        )
        cumulative_sine = torch.cumsum(
            injected_amplitude * inverse_magnitude * torch.sin(injection_phase),
            dim=1,
        )

        magnitude = torch.exp(log_magnitude)
        phase_cosine = torch.cos(cumulative_phase)
        phase_sine = torch.sin(cumulative_phase)
        cosine_state = magnitude * (
            cumulative_cosine * phase_cosine - cumulative_sine * phase_sine
        )
        sine_state = magnitude * (
            cumulative_cosine * phase_sine + cumulative_sine * phase_cosine
        )
        return torch.cat((cosine_state, sine_state), dim=-1)

    def forward(self, hidden: torch.Tensor, delta_t_raw: torch.Tensor) -> torch.Tensor:
        if delta_t_raw.ndim != 2 or delta_t_raw.shape[:2] != hidden.shape[:2]:
            raise ValueError("delta_t_raw must have shape [batch, obs_len]")
        params = self.wave_parameters()
        normalized = self.content_norm(hidden)
        drive = self.content_drive(normalized)
        content_gate = torch.sigmoid(self.content_gate(normalized))
        injected_amplitude = params["A"] * content_gate * drive
        delta_t = delta_t_raw.clamp_min(0.0).unsqueeze(-1)
        if self.scan_mode == "parallel":
            states = self._parallel_scan(injected_amplitude, delta_t, params)
        else:
            states = self._sequential_scan(injected_amplitude, delta_t, params)
        return self.wave_readout(states)


class WSPEncoderLayer(nn.Module):
    """Self-attention plus WSP, with no ReLU, GELU, or GLU feed-forward sublayer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        wave_scan: str = "sequential",
        use_phase_attention: bool = True,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.wave_norm = nn.LayerNorm(d_model)
        self.wave = WaveStatePropagation(d_model, scan_mode=wave_scan)
        self.fusion_gate = nn.Linear(d_model, d_model)
        self.wave_dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.phase_bias_scale = nn.Parameter(torch.full((nhead,), 0.1))
        self.use_phase_attention = use_phase_attention

        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

    def phase_attention_bias(
        self,
        delta_t_raw: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, sequence_length = delta_t_raw.shape
        params = self.wave.wave_parameters()
        k = params["k"].view(self.nhead, self.head_dim).mean(dim=-1)
        omega = params["omega"].view(self.nhead, self.head_dim).mean(dim=-1)
        theta = params["theta"].view(self.nhead, self.head_dim).mean(dim=-1)

        physical_time = torch.cumsum(delta_t_raw.clamp_min(0.0), dim=1)
        time_difference = physical_time.unsqueeze(2) - physical_time.unsqueeze(1)
        position = torch.arange(sequence_length, device=delta_t_raw.device, dtype=physical_time.dtype)
        position_difference = position.unsqueeze(1) - position.unsqueeze(0)
        phase = (
            k.view(1, self.nhead, 1, 1) * position_difference.view(1, 1, sequence_length, sequence_length)
            - omega.view(1, self.nhead, 1, 1) * time_difference.unsqueeze(1)
            + theta.view(1, self.nhead, 1, 1)
        )
        bias = self.phase_bias_scale.view(1, self.nhead, 1, 1) * torch.cos(phase)
        return bias.reshape(batch_size * self.nhead, sequence_length, sequence_length).to(dtype=dtype)

    def forward(self, hidden: torch.Tensor, delta_t_raw: torch.Tensor) -> torch.Tensor:
        propagated = self.wave(self.wave_norm(hidden), delta_t_raw)
        fusion = torch.sigmoid(self.fusion_gate(hidden))
        fused = hidden + self.wave_dropout(fusion * propagated)

        normalized = self.attention_norm(fused)
        phase_bias = (
            self.phase_attention_bias(delta_t_raw, normalized.dtype)
            if self.use_phase_attention
            else None
        )
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=phase_bias,
            need_weights=False,
        )
        return fused + self.attention_dropout(attended)


class WSPTrajectoryPredictor(nn.Module):
    """Transformer attention with elapsed-time wave hidden dynamics and a linear readout."""

    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        wave_scan: str = "sequential",
        use_phase_attention: bool = True,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position = LearnedPositionEmbedding(obs_len, d_model)
        self.layers = nn.ModuleList(
            WSPEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                wave_scan=wave_scan,
                use_phase_attention=use_phase_attention,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(obs_len * d_model, pred_len * 2)

    def forward(self, x: torch.Tensor, delta_t_raw: torch.Tensor) -> torch.Tensor:
        hidden = self.position(self.input_projection(x))
        for layer in self.layers:
            hidden = layer(hidden, delta_t_raw)
        hidden = self.final_norm(hidden)
        return self.readout(hidden.flatten(start_dim=1)).view(x.shape[0], self.pred_len, 2)

    def wave_parameter_summary(self) -> dict[str, float]:
        rows: dict[str, list[torch.Tensor]] = {}
        for layer in self.layers:
            for name, value in layer.wave.wave_parameters().items():
                rows.setdefault(name, []).append(value.detach().flatten())
        return {
            f"{name}_{stat}": float(getattr(torch.cat(values), stat)().cpu())
            for name, values in rows.items()
            for stat in ("mean", "std", "min", "max")
        }


def build_sequence_model(
    model_name: str,
    obs_len: int,
    pred_len: int,
    feature_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    ffn_dim: int,
    dropout: float,
) -> nn.Module:
    if model_name == "transformer":
        return TransformerTrajectoryPredictor(
            obs_len=obs_len,
            pred_len=pred_len,
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
    if model_name == "wsp":
        return WSPTrajectoryPredictor(
            obs_len=obs_len,
            pred_len=pred_len,
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown sequence model: {model_name}")


def sequence_model_version(model_name: str) -> str:
    if model_name == "transformer":
        return TRANSFORMER_MODEL_VERSION
    if model_name == "wsp":
        return WSP_MODEL_VERSION
    raise ValueError(f"Unknown sequence model: {model_name}")
