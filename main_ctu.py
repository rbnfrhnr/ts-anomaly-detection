import sys

import numpy as np
import torch

from augmentations.augmentation_factory import get_augmentation
from augmentations.vae_augmentation import VAEAugmentation
from data.ucr_loader import UCRDataset
from data.ctu_loader import CTUDataset
from downstream.downstream_factory import get_downstream
from evaluation.evaluator_factory import get_evaluator
from model.model_factory import get_model
from training.trainer_factory import get_trainer
from utils.common import read_cfg, setup_run
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    evaluator = cfg["evaluator"]
    augment_name = cfg["augmentation"]["name"] if "augmentation" in cfg else ""

    autoencoder = get_model(**cfg["autoencoder"])
    model = autoencoder.to(device)

    augmentation = get_augmentation(augment_name, **cfg)
    dataset = CTUDataset(**cfg["data"], cache_location=run_dir,
                         batch_transform=augmentation)

    norm_train, norm_test = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                    len(dataset) - int((0.8 * len(dataset)))])
    mal_train, mal_test = torch.utils.data.random_split(dataset.mal, [int(0.8 * len(dataset.mal)),
                                                                      len(dataset.mal) - int(
                                                                          (0.8 * len(dataset.mal)))])
    mal_train = torch.utils.data.TensorDataset(torch.Tensor(mal_train), torch.Tensor(np.ones_like(mal_train)))
    mal_test = torch.utils.data.TensorDataset(torch.Tensor(mal_test), torch.Tensor(np.ones_like(mal_test)))

    norm_train_loader = torch.utils.data.DataLoader(dataset=norm_train, batch_size=batch_size, shuffle=True)
    dtask_fit_loader = torch.utils.data.ConcatDataset([norm_train, mal_train])
    dtask_eval_loader = torch.utils.data.ConcatDataset([norm_test, mal_test])
    # dataset = UCRDataset(**cfg["data"], cache_location=run_dir)

    trainer = get_trainer(cfg["training"]["trainer_name"])
    trainer = trainer(model=autoencoder, device=device, run_dir=run_dir, **cfg["training"])
    autoencoder, losses = trainer.train(norm_train_loader)
    autoencoder.eval()
    dtask = get_downstream(downstream_cfg["type"], autoencoder, **downstream_cfg["config"]).fit(dtask_fit_loader)

    evaluator_fn = get_evaluator(evaluator)
    # evaluator_fn(model, dtask, dtask_eval_loader, dataset, run_dir=run_dir, **cfg)
    preds = dtask.predict(dtask_eval_loader)
    zeros = np.zeros(dtask_eval_loader.datasets[0].dataset[dtask_eval_loader.datasets[0].indices][0].shape[0])
    ones = np.ones(len(dtask_eval_loader.datasets[1].tensors))
    f1_score(np.concatenate([zeros, ones]), preds)
    print(preds)
