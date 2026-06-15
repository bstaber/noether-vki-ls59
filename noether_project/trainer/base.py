#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import torch
import torch.nn.functional as F
from noether.training.trainers import BaseTrainer
from noether.training.trainers.types import TrainerResult


class BoilerplateTrainer(BaseTrainer):
    """A base trainer implementation for the `Noether` framework that runs a simple forward pass and computes a loss value.
    This implementation overrides the `train_step` method to defined in the `BaseTrainer` class.
    However, one could also use the default implementation of the `BaseTrainer` class which performs a similar a forward pass and and the user needs to implement the `compute_loss` method instead.
    """

    @staticmethod
    def train_step(
        batch: dict[str, torch.Tensor], model: torch.nn.Module
    ) -> TrainerResult:
        """Forward method of the trainer that runs a forward pass on the model and computes the loss.

        Args:
            batch: dict with tensors for the forward pass and the loss computation.
            model: Model instance to run the forward pass on.

        Returns:
            TrainerResult containing the total loss.
        """
        # prepare data
        x = batch["x"]
        target = batch["y"]

        # forward
        y_hat = model(x)

        # calculate loss
        loss = F.cross_entropy(y_hat, target)

        return TrainerResult(total_loss=loss)


class UPTTrainer(BaseTrainer):
    @staticmethod
    def train_step(
        batch: dict[str, torch.Tensor], model: torch.nn.Module
    ) -> TrainerResult:
        """Forward method of the trainer that runs a forward pass on the model and computes the loss.

        Args:
            batch: dict with tensors for the forward pass and the loss computation.
            model: Model instance to run the forward pass on.

        Returns:
            TrainerResult containing the total loss.
        """
        geometry_position = batch["geometry_position"]
        geometry_batch_idx = batch.get("geometry_batch_idx")
        if geometry_batch_idx is None:
            # Fallback for default collate / batch_size=1 experiments.
            if geometry_position.ndim == 3:
                bsz, npts, _ = geometry_position.shape
                geometry_position = geometry_position.reshape(bsz * npts, -1)
                geometry_batch_idx = torch.arange(
                    bsz, device=geometry_position.device
                ).repeat_interleave(npts)
            else:
                geometry_batch_idx = torch.zeros(
                    geometry_position.shape[0],
                    dtype=torch.long,
                    device=geometry_position.device,
                )

        geometry_supernode_idx = batch["geometry_supernode_idx"]
        if geometry_supernode_idx.ndim == 2:
            bsz, n_super = geometry_supernode_idx.shape
            if bsz == 1:
                geometry_supernode_idx = geometry_supernode_idx[0]
            else:
                offsets = (
                    torch.arange(
                        bsz,
                        device=geometry_supernode_idx.device,
                        dtype=geometry_supernode_idx.dtype,
                    )
                    * (geometry_position.shape[0] // bsz)
                ).unsqueeze(1)
                geometry_supernode_idx = (geometry_supernode_idx + offsets).reshape(-1)

        prediction = model(
            geometry_batch_idx=geometry_batch_idx,
            geometry_supernode_idx=geometry_supernode_idx,
            geometry_position=geometry_position,
            query_position=batch["query_position"],
            in_scalars=batch["in_scalars"],
            unbatch_mask_query_position=batch.get("unbatch_mask_query_position"),
        )

        target = batch["target_mach"]
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        # Align target shape to model output:
        # - dense mode: prediction is [B, N, C], target should match
        # - sparse/unbatched mode: prediction is [B*N_valid, C], flatten target
        if prediction.ndim == 2 and target.ndim == 3:
            target = target.reshape(-1, target.shape[-1])
            unbatch_mask = batch.get("unbatch_mask_query_position")
            if unbatch_mask is not None:
                target = target[unbatch_mask]
        elif prediction.ndim == 3 and target.ndim == 2:
            target = target.unsqueeze(-1)

        loss = F.mse_loss(prediction, target)

        return TrainerResult(total_loss=loss, losses_to_log={"mse": loss})


class TransolverTrainer(BaseTrainer):
    @staticmethod
    def train_step(
        batch: dict[str, torch.Tensor], model: torch.nn.Module
    ) -> TrainerResult:

        geometry_position = batch["geometry_position"]

        prediction = model(
            x=geometry_position,
            condition=batch["in_scalars"],
        )

        target = batch["target_mach"]
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        loss = F.mse_loss(prediction, target)

        return TrainerResult(total_loss=loss, losses_to_log={"mse": loss})
