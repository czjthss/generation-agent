from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "AustraliaRainfall": "#2878B5",
    "BeijingPM25Quality": "#D95319",
    "BenzeneConcentration": "#3A923A",
}

LABELS = {
    "AustraliaRainfall": "AustraliaRainfall",
    "BeijingPM25Quality": "BeijingPM25Quality",
    "BenzeneConcentration": "BenzeneConcentration",
}


def _draw(ax, frame: pd.DataFrame, domain: str) -> None:
    color = COLORS[domain]
    ax.plot(frame["index"], frame["value"], color=color, linewidth=1.15)
    abnormal = frame["anomaly"] == 1
    if abnormal.any():
        ax.scatter(
            frame.loc[abnormal, "index"],
            frame.loc[abnormal, "value"],
            s=9,
            color="#C9252D",
            alpha=0.75,
            label="anomaly",
            zorder=3,
        )
    ax.set_ylabel("value", fontsize=14)
    ax.set_title(LABELS[domain], fontsize=16, fontweight="bold")
    ax.grid(True, color="#D9DEE5", linewidth=0.6, alpha=0.75)
    ax.tick_params(labelsize=11)
    ax.margins(x=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "samples_manifest.json").read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    frames = {}
    for sample in manifest["samples"]:
        domain = sample["domain"]
        frame = pd.read_csv(input_dir / sample["csv_file"])
        frames[domain] = frame
        fig, ax = plt.subplots(figsize=(12, 4.4))
        _draw(ax, frame, domain)
        ax.set_xlabel("index")
        fig.tight_layout()
        fig.savefig(output_dir / f"{domain}_typical.png", bbox_inches="tight")
        plt.close(fig)

    order = ["AustraliaRainfall", "BeijingPM25Quality", "BenzeneConcentration"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, domain in zip(axes, order):
        _draw(ax, frames[domain], domain)
    axes[-1].set_xlabel("index", fontsize=14)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(output_dir / "typical_samples_overview.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
