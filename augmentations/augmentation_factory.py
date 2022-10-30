from augmentations.vae_augmentation import VAEAugmentation


def get_augmentation(name, **cfg):
    if name == "vae_generate":
        return VAEAugmentation(**cfg)

    return None
