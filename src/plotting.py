from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_prediction_plot(pred: torch.Tensor, target: torch.Tensor, output_path: str | Path, n: int = 12) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred[:n].numpy()
    target = target[:n].numpy()

    cols = 4
    rows = max(1, (len(pred) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, ax in enumerate(axes.flat[: len(pred)]):
        ax.axis("on")
        ax.plot(target[i, :, 0], target[i, :, 1], marker="o", label="GT", linewidth=1.5)
        ax.plot(pred[i, :, 0], pred[i, :, 1], marker="x", label="Pred", linewidth=1.5)
        ax.set_title(f"sample {i}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.4)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
