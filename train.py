import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import DiT
from config import Config
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def save_images(cfg: Config, dataset: TensorDataset, generated: List[np.ndarray], epoch: int, num_samples: int):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    train_imgs = dataset[indices][0].permute(0, 2, 3, 1).numpy()

    gen_imgs = [
        (g.transpose(0, 2, 3, 1).astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        for g in generated]

    fig, axes = plt.subplots(1 + len(gen_imgs), num_samples, figsize=(num_samples * 3, 3*(1+len(gen_imgs))))
    for i in range(num_samples):
        axes[0, i].imshow(train_imgs[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Training Data', fontsize=12, loc='left')
        for j in range(1, len(gen_imgs) + 1):
            axes[j, i].imshow(gen_imgs[j - 1][i])
            axes[j, i].axis('off')
            if i == 0:
                axes[j, i].set_title(f'Generated {j}', fontsize=12, loc='left')

    plt.suptitle(f'Epoch {epoch}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{cfg.plot_dir}/comparison_{epoch:03d}.png', dpi=120)
    plt.close(fig)

def get_decay_params(model: nn.Module, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim < 2 or name.endswith(".bias") or name == "pos_embed":
                no_decay_params.append(param)
                # print('!', name)
            else:
                decay_params.append(param)
                # print('#', name)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

def train(cfg: Config):
    torch.set_float32_matmul_precision('high')

    if cfg.save_ckpt_interval > 0 and cfg.ckpt_dir:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.save_plot_interval > 0 and cfg.plot_dir:
        os.makedirs(cfg.plot_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = DiT(cfg).to(device)
    optimizer = optim.AdamW(get_decay_params(model, cfg.weight_decay), lr=cfg.lr, fused=device.type == "cuda")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    images = torch.from_numpy(np.load(cfg.data_path)).permute(0, 3, 1, 2).contiguous()
    dataset = TensorDataset(images)

    # 2 batch for validation
    num_val = cfg.batch_size * 2
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False, drop_last=False)
    print(f'TrainingData loaded; {len(train_dataset)} images in {len(train_loader)} batches')
    print(f'Validation data loaded; {len(val_dataset)} images in {len(val_loader)} batches')

    def calc_loss(y: torch.Tensor):
        B, C, H, W = y.shape
        y = y.to(device, non_blocking=True).float() / 255.0
        z = torch.randn(B, C, H, W, device=device)
        t = torch.rand(B, device=device)
        v = y - z
        x = z + v * t.view(B, 1, 1, 1)
        v_pred = model(x, t)
        loss = F.mse_loss(v_pred, v)
        return loss

    calc_loss_c = torch.compile(calc_loss)

    for epoch in range(cfg.epochs):
        total_loss = 0

        model.train()
        for (y,) in train_loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = calc_loss_c(y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        val_total_loss = 0
        model.eval()
        with torch.no_grad():
            for (y, ) in val_loader:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = calc_loss_c(y)
                    val_total_loss += loss.item()
        avg_val_loss = val_total_loss / len(val_loader)
        
        print(f"Epoch {epoch:03d}/{cfg.epochs} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if cfg.save_ckpt_interval > 0 and cfg.ckpt_dir and (epoch % cfg.save_ckpt_interval == 0 or epoch == cfg.epochs - 1):
            torch.save(model.state_dict(), f"{cfg.ckpt_dir}/epoch_{epoch:03d}.pt")

        if cfg.save_plot_interval > 0 and cfg.plot_dir and (epoch % cfg.save_plot_interval == 0 or epoch == cfg.epochs - 1):
            model.eval()
            num_samples = 8
            steps = 30
            x = torch.randn(num_samples, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)
            procs = []
            dt = 1.0 / steps
            with torch.no_grad():
                for i in range(steps):
                    t = torch.ones(num_samples, device=device) * i * dt
                    v_pred = model(x, t)
                    x = x + v_pred * dt
                    procs.append(x.cpu().clone().numpy())
            print(f"Generated {num_samples} samples")
            save_images(cfg, dataset, procs, epoch, num_samples)

    print("Training complete!")

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
