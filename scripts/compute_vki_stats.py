from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from plaid.storage import init_from_disk


def _to_feature_rows(x: torch.Tensor, key: str) -> torch.Tensor:
    """Convert tensor to [num_rows, num_features] for running moments."""
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1) if key == "in_scalars" else x.reshape(-1, 1)
    return x.reshape(-1, x.shape[-1])


def compute_stats(dataset, converter, num_samples: int) -> dict[str, list[float]]:
    keys = ["nodes", "mach", "in_scalars"]
    sums: dict[str, torch.Tensor] = {}
    sums_sq: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    for i in range(num_samples):
        sample = converter.sample_to_plaid(dataset[i])
        raw = {
            "nodes": torch.tensor(
                sample.get_nodes(base_name="Base_2_2"), dtype=torch.float64
            ),
            "mach": torch.tensor(
                sample.get_field(base_name="Base_2_2", name="mach"), dtype=torch.float64
            ),
            "in_scalars": torch.tensor(
                [
                    sample.get_scalar(name="angle_in"),
                    sample.get_scalar(name="mach_out"),
                ],
                dtype=torch.float64,
            ),
        }

        for key in keys:
            rows = _to_feature_rows(raw[key], key)

            if key not in sums:
                sums[key] = rows.sum(dim=0)
                sums_sq[key] = (rows * rows).sum(dim=0)
                counts[key] = rows.shape[0]
            else:
                sums[key] += rows.sum(dim=0)
                sums_sq[key] += (rows * rows).sum(dim=0)
                counts[key] += rows.shape[0]

    stats: dict[str, list[float]] = {}
    for key in keys:
        n = max(counts[key], 1)
        mean = sums[key] / n
        var = sums_sq[key] / n - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        stats[f"{key}_mean"] = mean.tolist()
        stats[f"{key}_std"] = std.tolist()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute VKI-LS59 normalization stats via Plaid only and save to YAML."
    )
    parser.add_argument(
        "--dataset-folder", required=True, help="Path to VKI-LS59 dataset folder"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output",
        default=str(Path("noether_project/datasets/vki_ls59_stats.yaml")),
        help="Output YAML file path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional number of samples to use (defaults to full split length).",
    )
    args = parser.parse_args()

    dataset_dict, converter_dict = init_from_disk(args.dataset_folder)
    dataset = dataset_dict[args.split]
    converter = converter_dict[args.split]

    split_len = len(dataset)
    num_samples = (
        split_len if args.num_samples is None else min(args.num_samples, split_len)
    )

    stats = compute_stats(dataset, converter, num_samples)
    keys = ["nodes", "mach", "in_scalars"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(stats, f, sort_keys=True)

    print(f"Saved VKI stats to: {output_path}")
    print(f"Computed from split='{args.split}' with {num_samples}/{split_len} samples")
    for key in keys:
        print(f"- {key}: mean={stats[f'{key}_mean']}, std={stats[f'{key}_std']}")


if __name__ == "__main__":
    main()
