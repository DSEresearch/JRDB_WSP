from __future__ import annotations

import math

import torch
import torch.nn as nn

from .wsp_review_models import PersistentWaveLayer, WSPAblation, time_features


MODEL_NAMES = ("wsp", "time_gru", "time_transformer")


class _ActivityPredictorBase(nn.Module):
    """Common input projection and classification head for activity models.

    At retained step ``t``, ``x[:, t]`` is the current relational scene state
    and ``delta_t[:, t]`` is the known outgoing gap.  The returned logits at
    step ``t`` predict the activity at the next retained observation.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        time_scale: float,
        d_model: int,
        max_length: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        if d_model <= 0 or max_length <= 0:
            raise ValueError("d_model and max_length must be positive")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.time_scale = float(time_scale)
        self.d_model = int(d_model)
        self.max_length = int(max_length)

        # Every comparison model uses the same projection and head design.
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, input_dim]")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {x.shape[-1]}"
            )
        if delta_t.ndim != 2 or delta_t.shape != x.shape[:2]:
            raise ValueError("delta_t must have shape [batch, time]")
        if valid_mask.ndim != 2 or valid_mask.shape != x.shape[:2]:
            raise ValueError("valid_mask must have shape [batch, time]")
        if x.shape[1] > self.max_length:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds max_length={self.max_length}"
            )

        mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if delta_t.device != x.device:
            delta_t = delta_t.to(x.device)
        safe_x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        safe_delta_t = torch.where(
            mask,
            delta_t.to(dtype=x.dtype),
            torch.zeros_like(delta_t, device=x.device, dtype=x.dtype),
        )
        if not torch.isfinite(safe_x).all():
            raise ValueError("x contains a non-finite value at a valid step")
        if not torch.isfinite(safe_delta_t).all():
            raise ValueError("delta_t contains a non-finite value at a valid step")
        safe_delta_t = safe_delta_t.clamp_min(0.0)
        return safe_x, safe_delta_t, mask

    def _classification_logits(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(self.output_norm(hidden))
        return torch.where(valid_mask.unsqueeze(-1), logits, torch.zeros_like(logits))


class WSPActivityPredictor(_ActivityPredictorBase):
    """Persistent WSP temporal operator followed by an activity head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        time_scale: float,
        d_model: int = 128,
        num_layers: int = 2,
        condition_rank: int = 16,
        max_length: int = 128,
    ) -> None:
        super().__init__(input_dim, num_classes, time_scale, d_model, max_length)
        if num_layers <= 0 or condition_rank <= 0:
            raise ValueError("num_layers and condition_rank must be positive")
        self.num_layers = int(num_layers)
        self.condition_rank = int(condition_rank)
        self.layers = nn.ModuleList(
            PersistentWaveLayer(
                self.d_model,
                self.condition_rank,
                omega_max=2.0 * math.pi,
                ablation=WSPAblation(),
            )
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x, delta_t, valid_mask = self._prepare_inputs(x, delta_t, valid_mask)
        batch, steps, _ = x.shape
        carry = x.new_zeros(batch, self.num_layers, self.d_model, 2)
        content = x.new_zeros(batch, self.d_model)
        outputs: list[torch.Tensor] = []

        for step in range(steps):
            step_valid = valid_mask[:, step]
            content_mask = step_valid.unsqueeze(-1)
            carry_mask = content_mask.unsqueeze(-1)
            observed = self.input_projection(x[:, step])
            # A padded step neither replaces the last content state nor changes
            # any layer's persistent cosine/sine carry.
            content = torch.where(content_mask, observed, content)
            tau = delta_t[:, step] / self.time_scale
            next_carries: list[torch.Tensor] = []
            for layer_index, layer in enumerate(self.layers):
                candidate_content, candidate_carry = layer.step(
                    content,
                    tau,
                    carry[:, layer_index],
                )
                content = torch.where(content_mask, candidate_content, content)
                next_carries.append(
                    torch.where(
                        carry_mask,
                        candidate_carry,
                        carry[:, layer_index],
                    )
                )
            carry = torch.stack(next_carries, dim=1)
            outputs.append(content)

        hidden = torch.stack(outputs, dim=1)
        return self._classification_logits(hidden, valid_mask)


class TimeAwareGRUActivityPredictor(_ActivityPredictorBase):
    """Stacked GRU baseline conditioned on the same outgoing physical gap."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        time_scale: float,
        d_model: int = 128,
        num_layers: int = 2,
        condition_rank: int = 16,
        max_length: int = 128,
    ) -> None:
        del condition_rank
        super().__init__(input_dim, num_classes, time_scale, d_model, max_length)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        self.time_projection = nn.Linear(10, self.d_model)
        self.cells = nn.ModuleList(
            nn.GRUCell(self.d_model, self.d_model)
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x, delta_t, valid_mask = self._prepare_inputs(x, delta_t, valid_mask)
        batch, steps, _ = x.shape
        hidden = [x.new_zeros(batch, self.d_model) for _ in self.cells]
        temporal = self.time_projection(time_features(delta_t, self.time_scale))
        projected = self.input_projection(x) + temporal
        outputs: list[torch.Tensor] = []

        for step in range(steps):
            step_valid = valid_mask[:, step].unsqueeze(-1)
            layer_input = projected[:, step]
            for layer_index, cell in enumerate(self.cells):
                candidate = cell(layer_input, hidden[layer_index])
                hidden[layer_index] = torch.where(
                    step_valid,
                    candidate,
                    hidden[layer_index],
                )
                layer_input = hidden[layer_index]
            outputs.append(hidden[-1])

        return self._classification_logits(torch.stack(outputs, dim=1), valid_mask)


class TimeConditionedTransformerActivityPredictor(_ActivityPredictorBase):
    """Causal Transformer baseline with explicit physical-time features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        time_scale: float,
        d_model: int = 128,
        num_layers: int = 2,
        condition_rank: int = 16,
        max_length: int = 128,
    ) -> None:
        del condition_rank
        super().__init__(input_dim, num_classes, time_scale, d_model, max_length)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        nhead = next(head for head in (8, 4, 2, 1) if self.d_model % head == 0)
        self.time_projection = nn.Linear(10, self.d_model)
        self.position = nn.Embedding(self.max_length, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=2 * self.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x, delta_t, valid_mask = self._prepare_inputs(x, delta_t, valid_mask)
        batch, steps, _ = x.shape
        positions = torch.arange(steps, device=x.device)
        hidden = (
            self.input_projection(x)
            + self.time_projection(time_features(delta_t, self.time_scale))
            + self.position(positions).unsqueeze(0)
        )
        hidden = torch.where(valid_mask.unsqueeze(-1), hidden, torch.zeros_like(hidden))

        # A fully padded row would otherwise present attention with no valid key.
        # Unmask a zero-valued sentinel solely for numerical safety; its logits
        # remain zero because the original valid_mask is applied after the head.
        key_padding_mask = ~valid_mask
        all_invalid = ~valid_mask.any(dim=1)
        if bool(all_invalid.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return self._classification_logits(encoded, valid_mask)


def build_activity_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    time_scale: float,
    d_model: int = 128,
    num_layers: int = 2,
    condition_rank: int = 16,
    max_length: int = 128,
) -> nn.Module:
    """Build one of the supported next-observation activity classifiers."""

    builders = {
        "wsp": WSPActivityPredictor,
        "time_gru": TimeAwareGRUActivityPredictor,
        "time_transformer": TimeConditionedTransformerActivityPredictor,
    }
    try:
        model_type = builders[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown activity model {model_name!r}; choose from {MODEL_NAMES}"
        ) from error
    return model_type(
        input_dim=input_dim,
        num_classes=num_classes,
        time_scale=time_scale,
        d_model=d_model,
        num_layers=num_layers,
        condition_rank=condition_rank,
        max_length=max_length,
    )
