import numpy as np
import torch
from scipy.signal import savgol_filter
from augmentations.augmentation_base import BaseAugmentation
import math
from torch.utils.data import DataLoader, TensorDataset


class VAEAugmentation(BaseAugmentation):

    def __init__(self, **cfg):
        super(VAEAugmentation, self).__init__(**cfg)
        self.vae_path = self.augment_cfg["generator_location"]
        self.device = cfg["torch-device"]
        self.std = self.augment_cfg["std"]
        self.smooth = self.augment_cfg["smooth"] if "smooth" in self.augment_cfg else True
        self.min_filter_window = self.augment_cfg[
            "min_filter_window"] if "min_filter_window" in self.augment_cfg else 40
        self.model = torch.load(self.vae_path)
        self.t_steps = cfg["data"]["t_steps"]
        self.model.eval()

    def augment(self, sample, *args, **kwargs):
        xx = torch.Tensor(sample).to(self.device)
        dataset = TensorDataset(xx, torch.ones(xx.shape[0]))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=512,
                                             shuffle=True)
        augmented = []
        for x, _ in loader:
            z = self.model.encoder(x)
            means = torch.zeros((x.shape[0], self.model.latent_dim)).to(self.device)
            stdv = torch.zeros((x.shape[0], self.model.latent_dim)).to(self.device) + self.std
            new_z = z + torch.normal(mean=means, std=stdv)
            new_x = self.model.decoder(new_z)
            new_x = new_x.detach().cpu().numpy()
            augmented.append(new_x)
        augmented = np.concatenate(augmented)
        if self.smooth:
            filter_window = min(self.min_filter_window, math.floor((augmented.shape[1] * 0.5)))
            polyorder = min(filter_window - 1, 7)
            augmented = savgol_filter(augmented.reshape(-1, self.t_steps), filter_window, polyorder)
            return augmented.reshape(sample.shape)

        return augmented
