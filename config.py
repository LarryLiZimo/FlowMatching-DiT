from dataclasses import dataclass

@dataclass
class Config:

    # Model
    hidden_dim: int = 512
    emb_hidden_dim:int = 128
    num_heads: int = 8
    num_layers: int = 6
    patch_size: int = 4
    dropout: float = 0.1
    
    # Training
    data_path: str = "./cat_dataset.npy"
    batch_size: int = 256
    lr: float = 8e-4
    weight_decay: float = 1e-2
    epochs: int = 200

    ckpt_dir: str = './ckpt'
    save_ckpt_interval: int = 100 # set to 0 to disable

    plot_dir: str = './plots'
    save_plot_interval: int = 10 # set to 0 to disable
    

    # You should not change the following parameters
    #   because the cat dataset has a fixed size of 64x64x3
    img_size: int = 64
    in_channels: int = 3

    