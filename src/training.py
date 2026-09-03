from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import metrics_from_totals, trajectory_metric_totals
from .models import WFBTrajectoryPredictor, laplacian_penalty


def forward_model(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    x = batch["x"].to(device, non_blocking=True)
    dt = batch["delta_t"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    if isinstance(model, WFBTrajectoryPredictor):
        pred, aux = model(x, dt)
    else:
        pred = model(x, dt)
        aux = {}
    return pred, target, aux


def temporal_correction_grads(
    model: WFBTrajectoryPredictor,
    aux: dict[str, torch.Tensor],
    lambda_laplacian: float,
) -> tuple[tuple[nn.Parameter, ...], tuple[torch.Tensor, ...]]:
    """Differentiate the component-wise curvature correction only for temporal parameters.

    This is deliberately a custom gradient branch rather than the gradient of a
    global ``task_loss + penalty`` objective: k, Ax, theta_x, and the projection
    remain governed by the supervised task loss.
    """
    parameters = (model.wfb.At_raw, model.wfb.omega_raw, model.wfb.theta_t)
    correction = lambda_laplacian * laplacian_penalty(aux)
    gradients = torch.autograd.grad(correction, parameters, retain_graph=True)
    return parameters, gradients


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_laplacian: float,
    grad_clip: float,
    wfb_variant: str = "standard",
    warmup: bool = False,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    loss_sum = 0.0
    sample_count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        pred, target, aux = forward_model(model, batch, device)
        loss = criterion(pred, target)
        if isinstance(model, WFBTrajectoryPredictor) and aux and wfb_variant in {"laplacian", "combined"} and not warmup:
            temporal_parameters, correction_grads = temporal_correction_grads(
                model, aux, lambda_laplacian=lambda_laplacian
            )
            loss.backward()
            for parameter, correction_grad in zip(temporal_parameters, correction_grads):
                if wfb_variant == "laplacian":
                    parameter.grad = correction_grad.detach().clone()
                else:
                    parameter.grad.add_(correction_grad.detach())
        else:
            loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(target.shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    return loss_sum / max(1, sample_count)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "squared_error_sum": 0.0,
        "coordinate_count": 0.0,
        "displacement_sum": 0.0,
        "displacement_count": 0.0,
        "final_displacement_sum": 0.0,
        "sample_count": 0.0,
    }
    for batch in tqdm(loader, desc="eval", leave=False):
        pred, target, _ = forward_model(model, batch, device)
        for key, value in trajectory_metric_totals(pred, target).items():
            totals[key] += value
    return metrics_from_totals(totals)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, targets = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        pred, target, _ = forward_model(model, batch, device)
        preds.append(pred.cpu())
        targets.append(target.cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def save_history(history: list[dict[str, float]], path: str | Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)
