from augmentations.vae_augmentation import VAEAugmentation
from augmentations.reverse_augmentation import ReverseAugemnation
from augmentations.gaussian_noise_augmentation import GaussianNoiseAugmentation
from augmentations.replicate_augmentation import ReplicateAugemnation


def get_augmentation(name, **cfg):
    if name == "vae_generate":
        return VAEAugmentation(**cfg)

    if name == "reverse":
        return ReverseAugemnation(**cfg)

    if name == "noise":
        return GaussianNoiseAugmentation(**cfg)

    if name == "replicate":
        return ReplicateAugemnation(**cfg)

    return None
