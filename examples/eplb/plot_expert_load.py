#!/usr/bin/env python3
"""Plot a histogram of the logical expert load trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt

import numpy as np



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the histogram of logical expert load values."
    )
    default_data = Path(__file__).resolve().parent / "data" / "expert-load.json"
    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help=f"Path to expert load JSON (default: {default_data})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (defaults to <data-stem>-hist.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def _flatten_numbers(values: Sequence) -> Iterator[float]:
    for value in values:
        if isinstance(value, (int, float)):
            yield float(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            yield from _flatten_numbers(value)


def load_expert_values(path: Path):
    with path.open() as fp:
        payload = json.load(fp)

    history = payload.get("load_history")
    if not isinstance(history, list):
        raise ValueError("Expected 'load_history' to be a list.")

    if np is not None:
        arrays: list[np.ndarray] = []
        for entry in history:
            loads = entry.get("logical_expert_load") if isinstance(entry, dict) else None
            if not loads:
                continue
            arrays.append(np.asarray(loads, dtype=np.float64).ravel())
        if not arrays:
            raise ValueError("No load values found in JSON.")
        return np.concatenate(arrays)

    flattened: list[float] = []
    for entry in history:
        loads = entry.get("logical_expert_load") if isinstance(entry, dict) else None
        if not loads:
            continue
        flattened.extend(_flatten_numbers(loads))

    if not flattened:
        raise ValueError("No load values found in JSON.")

    return flattened



def plot_histogram(values: np.ndarray, bins: int, output: Path, show: bool) -> None:
    print(values.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, color="red", edgecolor="black", log=True)
    ax.set_xlabel("Logical expert load")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(
        f"Logical expert load distribution\nN={values.size:,}, bins={bins}"
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()

    fig.savefig(output, dpi=300)
    print(f"Saved histogram to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    values = load_expert_values(args.data)
    bins = 1000

    output = (
        args.output
        if args.output is not None
        else args.data.with_name(f"{args.data.stem}-hist.png")
    )

    print(
        f"Loaded {values.size:,} values "
        f"(min={values.min():.2f}, max={values.max():.2f}, bins={bins})"
    )
    plot_histogram(values, bins, output, args.show)


if __name__ == "__main__":
    main()