from augmentations.vae_augmentation import VAEAugmentation
from augmentations.reverse_augmentation import ReverseAugemnation
from augmentations.gaussian_noise_augmentation import GaussianNoiseAugmentation


def get_augmentation(name, **cfg):
    if name == "vae_generate":
        return VAEAugmentation(**cfg)

    if name == "reverse":
        return ReverseAugemnation(**cfg)

    if name == "noise":
        return GaussianNoiseAugmentation(**cfg)

    return None
