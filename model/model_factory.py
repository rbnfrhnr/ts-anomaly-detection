from model.vautoencoder import VariationalAutoencoder
from model.autoencoder import Autoencoder


def get_model(type=None, **cfg):
    if type.lower() == "ae":
        return Autoencoder(**cfg)
    if type.lower() == "vae":
        return VariationalAutoencoder(**cfg)
