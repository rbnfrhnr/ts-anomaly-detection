import sys

import torch

from data.ucr_loader import UCRDataset
from model.model_factory import get_model
from training.trainer_factory import get_trainer
from utils.common import read_cfg, setup_run

if __name__ == '__main__':
    cfg_file = sys.argv[1]
    cfg = read_cfg(cfg_file)
    cfg = setup_run(**cfg)
    device = cfg["torch-device"]
    run_dir = cfg["run-dir"]
    train_cfg = cfg["training"]
    epochs = train_cfg["epochs"]
    downstream_cfg = cfg["downstream"]
    window_size = cfg["data"]["t_steps"]
    batch_size = train_cfg["batch_size"]

    autoencoder = get_model(**cfg["autoencoder"])
    model = autoencoder.to(device)

    dataset = UCRDataset(**cfg["data"], cache_location=run_dir)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

    trainer = get_trainer(cfg["training"]["trainer_name"])
    trainer = trainer(model=autoencoder, device=device, run_dir=run_dir, **cfg["training"])
    autoencoder, losses = trainer.train(loader)
    autoencoder.eval()
