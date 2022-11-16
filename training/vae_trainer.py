import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
import math
import numpy as np
from copy import deepcopy, copy
from training.base_trainer import BaseTrainer


class VAETrainer(BaseTrainer):

    def __init__(self, kl_weight=0.5, **kwargs):
        super(VAETrainer, self).__init__(**kwargs)
        self.kl_weight = kl_weight
        self.loss_hist = []
        self.kl_hist = []
        self.total_loss_hist = []
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        print(self.kl_weight)

    def train_fn(self, loader):
        n_batches = math.ceil(len(loader.dataset) / loader.batch_size)
        columns = ["mse_loss", "kl_loss", "loss", "step", "epoch"]
        log_f = pd.DataFrame(columns=columns, data=[])
        best_loss = math.inf
        best_state_dict = None
        for epoch in range(self.epochs):
            with tqdm.tqdm(loader, position=0) as pbar:
                for (image, _) in loader:
                    image = image.to(self.device).float()

                    reconstructed = self.model(image)
                    kl_loss = self.model.encoder.kl
                    fn_loss = self.loss_fn(reconstructed, image[:,:,0].reshape(reconstructed.shape))
                    loss = (1 - self.kl_weight) * fn_loss + self.kl_weight * kl_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.loss_hist.append(fn_loss.detach().item())
                    self.kl_hist.append(kl_loss.detach().item())
                    self.total_loss_hist.append(loss.detach().item())
                    pbar.set_description_str("Epoch " + str(epoch))
                    pbar.set_postfix_str(
                        self.loss_fn._get_name() + ": " + str(fn_loss.detach().item()) + " kl-loss: " + str(
                            kl_loss.detach().item()) + " total-loss: " + str(self.total_loss_hist[-1]))
                    pbar.update(1)
                    data = [self.loss_hist[-1], self.kl_hist[-1], self.loss_hist[-1], len(self.loss_hist), epoch]
                    rec = pd.DataFrame(columns=columns, data=[data])
                    log_f = pd.concat([log_f, rec])

                if self.save_checkpoint and (epoch + 1) % self.checkpoint_interval == 0:
                    torch.save(self.model, self.checkpoint_dir + "/model-" + str(epoch + 1) + ".pkl")
                if self.save_best and np.mean(self.total_loss_hist[-n_batches:]) < best_loss:
                    best_state_dict = self.model.state_dict()
                    best_loss = np.mean(self.loss_hist[-n_batches:])

        if best_state_dict is not None:
            torch.save(best_state_dict, self.checkpoint_dir + "/model-best-state-dict.pkl")

        log_f.to_csv(self.run_dir + "/train_hist.csv")
        return self.model, (self.loss_hist, self.kl_hist)

    def infos(self):
        plt.style.use('fivethirtyeight')
        fig, axs = plt.subplots(3, 1, figsize=(15, 7))
        axs[0].set_title("info charts")
        axs[0].plot(self.loss_hist, label="MSE Loss")
        axs[0].legend()
        axs[1].plot(self.kl_hist, label="KL Loss")
        axs[1].legend()
        axs[2].plot(self.total_loss_hist, label="Total Loss")
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(self.run_dir + "/training_charts.png")
        plt.show()
