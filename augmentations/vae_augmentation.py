import torch
from scipy.signal import savgol_filter
import math


class VAEAugmentation(object):

    def __init__(self, augmentation={}, **cfg):
        super(VAEAugmentation, self).__init__()
        self.augment_cfg = augmentation["config"]
        self.vae_path = self.augment_cfg["generator_location"]
        self.device = cfg["torch-device"]
        self.std = self.augment_cfg["std"]
        self.model = torch.load(self.vae_path)
        self.t_steps = cfg["data"]["t_steps"]
        self.model.eval()

    def __call__(self, sample, *args, **kwargs):
        z = self.model.encoder(torch.Tensor(sample).to(self.device))
        means = torch.zeros((sample.shape[0], self.model.latent_dim)).to(self.device)
        stdv = torch.zeros((sample.shape[0], self.model.latent_dim)).to(self.device) + self.std
        new_z = z + torch.normal(mean=means, std=stdv)
        new_x = self.model.decoder(new_z)
        new_x = new_x.detach().cpu().numpy()
        new_x = savgol_filter(new_x.reshape(-1, self.t_steps), math.floor((new_x.shape[1] * 0.25)), 5)
        # new_x = new_x.reshape(*new_x.shape[1:])
        return new_x
