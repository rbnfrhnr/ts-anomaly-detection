import torch
from scipy.signal import savgol_filter
from augmentations.augmentation_base import BaseAugmentation
import math


class VAEAugmentation(BaseAugmentation):

    def __init__(self, **cfg):
        super(VAEAugmentation, self).__init__(**cfg)
        self.vae_path = self.augment_cfg["generator_location"]
        self.device = cfg["torch-device"]
        self.std = self.augment_cfg["std"]
        self.model = torch.load(self.vae_path)
        self.t_steps = cfg["data"]["t_steps"]
        self.model.eval()

    def augment(self, sample, *args, **kwargs):
        z = self.model.encoder(torch.Tensor(sample).to(self.device))
        means = torch.zeros((sample.shape[0], self.model.latent_dim)).to(self.device)
        stdv = torch.zeros((sample.shape[0], self.model.latent_dim)).to(self.device) + self.std
        new_z = z + torch.normal(mean=means, std=stdv)
        new_x = self.model.decoder(new_z)
        new_x = new_x.detach().cpu().numpy()
        filter_window = min(40, math.floor((new_x.shape[1] * 0.5)))
        polyorder = min(filter_window - 1, 7)
        new_x = savgol_filter(new_x.reshape(-1, self.t_steps), filter_window, polyorder)
        return new_x.reshape(sample.shape)
