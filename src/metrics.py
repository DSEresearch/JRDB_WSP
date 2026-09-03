from __future__ import annotations

import numpy as np
import torch


def trajectory_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute ADE/FDE/MSE/RMSE for tensors shaped [batch, pred_len, 2]."""
    with torch.no_grad():
        diff = pred - target
        sq = diff.pow(2)
        mse = sq.mean().item()
        rmse = float(np.sqrt(mse))
        disp = torch.linalg.norm(diff, dim=-1)
        ade = disp.mean().item()
        fde = disp[:, -1].mean().item()
    return {"ADE": ade, "FDE": fde, "MSE": mse, "RMSE": rmse}


def trajectory_metric_totals(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return additive totals so metrics remain exact across unequal batch sizes."""
    with torch.no_grad():
        diff = pred - target
        displacement = torch.linalg.norm(diff, dim=-1)
        return {
            "squared_error_sum": float(diff.pow(2).sum().item()),
            "coordinate_count": float(diff.numel()),
            "displacement_sum": float(displacement.sum().item()),
            "displacement_count": float(displacement.numel()),
            "final_displacement_sum": float(displacement[:, -1].sum().item()),
            "sample_count": float(displacement.shape[0]),
        }


def metrics_from_totals(totals: dict[str, float]) -> dict[str, float]:
    if totals.get("sample_count", 0.0) <= 0:
        return {"ADE": np.nan, "FDE": np.nan, "MSE": np.nan, "RMSE": np.nan}
    mse = totals["squared_error_sum"] / totals["coordinate_count"]
    return {
        "ADE": totals["displacement_sum"] / totals["displacement_count"],
        "FDE": totals["final_displacement_sum"] / totals["sample_count"],
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average already-complete run metrics (for seed-level summaries only)."""
    if not rows:
        return {"ADE": np.nan, "FDE": np.nan, "MSE": np.nan, "RMSE": np.nan}
    keys = ["ADE", "FDE", "MSE", "RMSE"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
