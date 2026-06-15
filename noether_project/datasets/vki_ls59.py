from pathlib import Path
from typing import Any, Literal

import torch
from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data import Dataset
from noether.data.base.dataset import with_normalizers
from plaid.containers import Sample
from plaid.storage import init_from_disk


class DatasetVKIConfig(DatasetBaseConfig):
    """Configuration for the VKI-LS59 dataset."""

    dataset_folder: str
    num_samples: int = 671
    num_supernodes: int = 1024
    split: Literal["train", "test"]
    supernode_sampling: Literal["deterministic", "uniform_random"] = "uniform_random"
    supernode_seed: int | None = None


class DatasetVKI(Dataset):
    """Dataset class for the VKI-LS59 dataset."""

    STATS_FILE: str = str(Path(__file__).with_name("vki_ls59_stats.yaml"))

    def __init__(
        self,
        dataset_config: DatasetVKIConfig,
    ):
        super().__init__(dataset_config=dataset_config)
        self.num_samples = dataset_config.num_samples
        self.num_supernodes = dataset_config.num_supernodes
        self.supernode_sampling = dataset_config.supernode_sampling
        self.supernode_seed = dataset_config.supernode_seed
        self.dataset_folder = dataset_config.dataset_folder
        self.split = dataset_config.split

        datasetdict, converterdict = init_from_disk(self.dataset_folder)

        self.dataset = datasetdict[self.split]
        self.converter = converterdict[self.split]

    def fetch_statistics(self) -> dict[str, list[float] | float] | None:
        stats_path = Path(self.STATS_FILE)
        if not stats_path.exists():
            return None
        return super().fetch_statistics()

    def __len__(self) -> int:
        return self.num_samples

    def pre_getitem(self, idx: int) -> dict[str, Any]:
        sample = self.converter.sample_to_plaid(self.dataset[idx])
        return {"sample": sample}

    @staticmethod
    def _get_nodes_raw(sample: Sample) -> torch.Tensor:
        return torch.tensor(
            sample.get_nodes(base_name="Base_2_2"),
            dtype=torch.float32,
        )

    @staticmethod
    def _get_mach_raw(sample: Sample) -> torch.Tensor:
        return torch.tensor(
            sample.get_field(base_name="Base_2_2", name="mach"),
            dtype=torch.float32,
        )

    # @with_normalizers("nodes")
    def getitem_nodes(self, idx: int, sample: Sample) -> torch.Tensor:
        return self._get_nodes_raw(sample)

    # @with_normalizers("mach")
    def getitem_mach(self, idx: int, sample: Sample) -> torch.Tensor:
        return self._get_mach_raw(sample)

    @with_normalizers("in_scalars")
    def getitem_in_scalars(self, idx: int, sample: Sample) -> torch.Tensor:
        angle_in = sample.get_scalar(name="angle_in")
        mach_out = sample.get_scalar(name="mach_out")
        return torch.tensor(
            [angle_in, mach_out],
            dtype=torch.float32,
        )

    # @with_normalizers("nodes")
    def getitem_geometry_position(self, idx: int, sample: Sample) -> torch.Tensor:
        return self._get_nodes_raw(sample)

    # @with_normalizers("nodes")
    def getitem_query_position(self, idx: int, sample: Sample) -> torch.Tensor:
        return self._get_nodes_raw(sample)

    def getitem_geometry_supernode_idx(self, idx: int, sample: Sample) -> torch.Tensor:
        num_nodes = int(sample.get_nodes(base_name="Base_2_2").shape[0])
        if num_nodes <= 0:
            return torch.empty(0, dtype=torch.long)

        k = min(self.num_supernodes, num_nodes)
        if k == num_nodes:
            return torch.arange(num_nodes, dtype=torch.long)

        if self.supernode_sampling == "uniform_random":
            if self.supernode_seed is None:
                supernode_idx = torch.randperm(num_nodes)[:k]
            else:
                generator = torch.Generator()
                generator.manual_seed(self.supernode_seed + idx)
                supernode_idx = torch.randperm(num_nodes, generator=generator)[:k]
            # Keep ascending order for stable neighborhood traversal downstream.
            return torch.sort(supernode_idx).values

        # Deterministic subsampling across the point-cloud extent.
        supernode_idx = torch.linspace(
            0, num_nodes - 1, steps=k, dtype=torch.float64
        ).long()

        supernode_idx = torch.unique_consecutive(supernode_idx)
        if supernode_idx.numel() < k:
            all_idx = torch.arange(num_nodes, dtype=torch.long)
            mask = torch.ones(num_nodes, dtype=torch.bool)
            mask[supernode_idx] = False
            extra = all_idx[mask][: (k - supernode_idx.numel())]
            supernode_idx = torch.cat([supernode_idx, extra], dim=0)
        return supernode_idx

    # @with_normalizers("mach")
    def getitem_target_mach(self, idx: int, sample: Sample) -> torch.Tensor:
        target = self._get_mach_raw(sample)
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        return target


if __name__ == "__main__":
    """Example usage of the DatasetVKI class."""

    config = DatasetVKIConfig(
        dataset_folder="/home/brian/data/physarena/VKI-LS59",
        dataset_normalizers={
            "nodes": {
                "kind": "noether.data.preprocessors.normalizers.FieldNormalizer",
                "strategy": "mean_std",
            },
            "mach": {
                "kind": "noether.data.preprocessors.normalizers.FieldNormalizer",
                "strategy": "mean_std",
            },
            "in_scalars": {
                "kind": "noether.data.preprocessors.normalizers.FieldNormalizer",
                "strategy": "mean_std",
            },
        },
        split="train",
    )
    ds = DatasetVKI(config)

    sample = ds.__getitem__(0)
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    sample = ds.__getitem__(1)
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
