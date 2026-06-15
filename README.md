### Noether applied to VKI-LS59 dataset (see [here](https://huggingface.co/datasets/PhysArena/VKI-LS59))

I added:
- a dataset class: `noether_project/datasets/vki_ls59.py` that loads samples from the VKI-LS59 through the the PLAID library
- a conditioned transolver model: `noether_project/models/conditioned_transolver.py`
- a conditioned UPT model: `noether_project/models/conditioned_upt.py` but not tested
- UPT and Transolver trainers in `noether_project/trainer/base.py`
- two scripts that download the dataset and compute some statistics in `noether_project/scripts/`

I mostly trained the Transolver model to check if I could reproduce the results obtained with, e.g., PhysicsNemo.

I run the training with the following command:

```bash
uv run noether-train --hp noether_project/configs/vki_transolver_experiment.yaml
```

Two examples of outptus are uploaded in `outputs`.