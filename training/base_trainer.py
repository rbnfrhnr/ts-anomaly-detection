import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import os
from pathlib import Path


class BaseTrainer():

    def __init__(self, model=None, epochs=150, device="cuda", infos=False, run_dir=None,
                 save_checkpoint=True, checkpoint_interval=10, save_best=True,
                 save_latest=True, **kwargs):
        self.model = model
        self.epochs = epochs
        self.device = device
        self.run_dir = run_dir

        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(model.parameters())
        self.show_info = infos
        self.save_checkpoint = save_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.save_best = save_best
        self.save_latest = save_latest
        self.checkpoint_dir = self.run_dir + "/model-checkpoints"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(self, loader):
        rv = self.train_fn(loader)
        if self.show_info:
            self.infos()
        if self.save_latest:
            torch.save(self.model, self.checkpoint_dir + "/model-final.pkl")
        return rv

    @abstractmethod
    def infos(self):
        raise NotImplementedError

    @abstractmethod
    def train_fn(self, loader):
        raise NotImplementedError
