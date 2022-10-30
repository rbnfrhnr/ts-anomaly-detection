from training.ae_trainer import AETrainer
from training.vae_trainer import VAETrainer


def get_trainer(name):
    if name == "ae_trainer":
        return AETrainer
    if name == "vae_trainer":
        return VAETrainer
