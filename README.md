## FlowMatching-DiT

Minimal PyTorch implementation of a Diffusion Transformer (DiT) trained with flow matching on `64x64` cat-face images.

## What This Repo Contains

- `model.py`: DiT model with patch embedding, sinusoidal time embedding, AdaLN conditioning, and transformer blocks.
- `train.py`: training loop, validation split, sampling, checkpoint saving, and plot generation.
- `config.py`: central training and model configuration.

## Dataset

The dataset used for this project is available on [Kaggle: larrylizimoccc/catfaces](https://www.kaggle.com/datasets/larrylizimoccc/catfaces/data).

Training expects a NumPy file at the path set by `Config.data_path`:

- default: `./cat_dataset.npy`
- shape: `(N, 64, 64, 3)`
- channel order: RGB

## Setup

Create an environment and install the required packages:

```bash
uv venv --python=3.11 # newer versions are fine

uv pip install torch numpy matplotlib
```

If you have not install uv, run:
```bash
pip install uv
```

If you want GPU training, install a CUDA-enabled PyTorch build that matches your system from the [official PyTorch install page](https://pytorch.org/get-started/locally/).

## Configuration

Edit `config.py` to change the training setup. Important fields:

- `data_path`: path to the `.npy` dataset
- `batch_size`: training batch size
- `lr`: learning rate
- `epochs`: number of epochs
- `ckpt_dir`: checkpoint output directory
- `save_ckpt_interval`: checkpoint save frequency
- `plot_dir`: generated image comparison output directory
- `save_plot_interval`: sampling / plot save frequency

Note: `img_size` and `in_channels` are fixed for the current dataset and should stay at `64` and `3`.

## Train

Run:

```bash
python train.py
```

During training the script:

- loads the dataset from `Config.data_path`
- reserves `2 * batch_size` samples for validation
- trains the DiT to predict the flow target with MSE loss
- saves checkpoints to `ckpt/`
- writes generated image comparisons to `plots/`

## Outputs

- checkpoints: `ckpt/epoch_XXX.pt`
- sample grids: `plots/comparison_XXX.png`

## Notes

- Mixed precision uses `bfloat16` autocast.
- The training step compiles the loss path with `torch.compile`.
- On an RTX 5090, training takes about 50 minutes.
