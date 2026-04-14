## FlowMatching-DiT

Minimal PyTorch implementation of a Diffusion Transformer (DiT) trained with flow matching on `64x64` cat-face images.

## Dataset

The dataset used for this project is available on [Kaggle: larrylizimoccc/catfaces](https://www.kaggle.com/datasets/larrylizimoccc/catfaces/data).

Download the dataset (unzip it possibly) and set the corresponding path in `config.py`.

## Setup

If you have not install uv, run:

```bash
pip install uv
```

Create an environment and install the required packages:

```bash
uv venv --python=3.13 # newer versions are fine

uv pip install torch numpy matplotlib
```

To start training, run:
```bash
python train.py
```

---

## Tutorial: Flow Matching

### Definitions

| Symbol | Meaning |
|---|---|
| **Y** | Target distribution — the dataset of real images |
| **y ~ Y** | A real image sampled from the dataset |
| **z ~ N(0, I)** | A noise image sampled from a standard Gaussian, same shape as `y` |
| **t** | A scalar timestep in `[0, 1]`. `t=0` is pure noise, `t=1` is a real image |
| **x_t** | The interpolated image at time `t`: `t` percent of the way from `z` to `y` |
| **v** | Velocity — the constant direction pointing from `z` to `y`, i.e. `v = y - z` |
| **model(x_t, t)** | Neural network that predicts `v` given the current image and timestep |

---

### Training

The idea: for every pair `(z, y)`, connect them with a straight line. Any point on that line is `x_t = z + (y - z) * t`. The velocity along a straight line is constant: `v = y - z`. We train the model to predict this velocity from any `(x_t, t)`.

**Algorithm:**

1. Sample `y ~ Y` from the dataset
2. Sample `z ~ N(0, I)`
3. Sample `t ~ Uniform(0, 1)`
4. Interpolate: `x_t = z + (y - z) * t`
5. Compute target velocity: `v = y - z`
6. Predict: `v_pred = model(x_t, t)`
7. Loss: `MSE(v_pred, v)` → backpropagate

```python
z = torch.randn_like(y)
t = torch.rand(B, device=device)
v = y - z
x_t = z + v * t.view(B, 1, 1, 1)
v_pred = model(x_t, t)
loss = F.mse_loss(v_pred, v)
```

---

### Sampling

The idea: start from `z ~ N(0, I)` at `t=0`. Repeatedly ask the model which direction to move, take a small step, and advance `t`. After `N` steps you arrive at `t=1` — a generated image.

**Algorithm:**

1. Sample initial noise `x ~ N(0, I)`
2. Set step size `dt = 1 / N`
3. For `i = 0, 1, ..., N-1`:
   1. Set `t = i * dt`
   2. Predict velocity: `v = model(x, t)`
   3. Step forward: `x = x + v * dt`
4. Return `x`

```python
x = torch.randn(B, C, H, W, device=device)
dt = 1.0 / steps
for i in range(steps):
    t = torch.ones(B, device=device) * i * dt
    v_pred = model(x, t)
    x = x + v_pred * dt
```

Because paths are straight, 30 steps is well enough. Diffusion models need hundreds because their paths are curved.

---

### The Model (DiT)

The model is a Diffusion Transformer (DiT). It takes `(x_t, t)` and returns a velocity image of the same shape. The image is patchified into a token sequence and fed into transformer blocks. The timestep `t` is sinusoidally encoded and used to modulate the layer normalization inside each block (AdaLN), so the model's behavior shifts depending on where along the path it is. A final linear layer maps the output tokens back to pixel space.
