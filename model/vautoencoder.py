import torch
import torch.nn as nn

import utils.torch as tutils


def create_layers(layers):
    created_layers = nn.ModuleList()
    for layer_cfg in layers:
        layer_name = layer_cfg["name"]
        layer_params = layer_cfg["config"] if "config" in layer_cfg else {}
        layer = getattr(nn, layer_name) if hasattr(nn, layer_name) else getattr(tutils, layer_name)
        layer = layer(**layer_params)
        created_layers.append(layer)
    return created_layers


class VariationalEncoder(nn.Module):

    def __init__(self, layers={}, latent_dim=5, **kwargs):
        super(VariationalEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layers = create_layers(layers)
        self.mu_layer = nn.Linear(in_features=self.layers[-1].out_features, out_features=latent_dim)
        self.sigma_layer = nn.Linear(in_features=self.layers[-1].out_features, out_features=latent_dim)
        self.normal = torch.distributions.Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()  # hack to get sampling on the GPU
        self.normal.scale = self.normal.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.layers[0](x)
        for idx, layer in enumerate(self.layers[1:]):
            x = layer(x)
        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))
        z = mu + sigma * self.normal.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).mean()
        return z

    def to(self, device):
        self.device = device
        self.layers = nn.ModuleList([layer.to(device) for layer in self.layers])
        self.mu_layer = self.mu_layer.to(device)
        self.sigma_layer = self.sigma_layer.to(device)
        return self


class Decoder(nn.Module):

    def __init__(self, layers={}, latent_dim=5, **kwargs):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.layers = create_layers(layers)

    def forward(self, x):
        y = self.layers[0](x)
        for idx, layer in enumerate(self.layers[1:]):
            y = layer(y)
        return y

    def to(self, device):
        self.device = device
        self.layers = nn.ModuleList([layer.to(device) for layer in self.layers])
        return self


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dim=10, **cfg):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_cfg = cfg["encoder"]
        self.decoder_cfg = cfg["decoder"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = VariationalEncoder(latent_dim=latent_dim, **self.encoder_cfg)
        self.decoder = Decoder(latent_dim=latent_dim, **self.decoder_cfg)

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        return self
