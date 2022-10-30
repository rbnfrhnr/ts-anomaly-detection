import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from training.base_trainer import BaseTrainer


class AETrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super(AETrainer, self).__init__(**kwargs)
        self.loss_hist = []

    def train_fn(self, loader):
        n_batches = math.ceil(len(loader.dataset) / loader.batch_size)
        columns = ["mse_loss", "step", "epoch"]
        log_f = pd.DataFrame(columns=columns, data=[])
        best_loss = np.inf
        best_state_dict = None
        self.loss_hist = []
        for epoch in range(self.epochs):
            with tqdm.tqdm(loader, position=0) as pbar:
                for (image, _) in loader:
                    image = image.to(self.device).float()

                    reconstructed = self.model(image)

                    loss = self.loss_fn(reconstructed, image)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.loss_hist.append(loss.detach().item())
                    pbar.set_description_str("Epoch " + str(epoch))
                    pbar.set_postfix_str(self.loss_fn._get_name() + ": " + str(loss.detach().item()))
                    pbar.update(1)
                    data = [self.loss_hist[-1], len(self.loss_hist), epoch]
                    rec = pd.DataFrame(columns=columns, data=[data])
                    log_f = pd.concat([log_f, rec])

            if self.save_checkpoint and (epoch + 1) % self.checkpoint_interval == 0:
                torch.save(self.model, self.checkpoint_dir + "/model-" + str(epoch + 1) + ".pkl")
            if self.save_best and np.mean(self.loss_hist[-n_batches:]) < best_loss:
                best_state_dict = self.model.state_dict()
                best_loss = np.mean(self.loss_hist[-n_batches:])

        if best_state_dict is not None:
            torch.save(best_state_dict, self.checkpoint_dir + "/model-best-state-dict.pkl")

        log_f.to_csv(self.run_dir + "/train_hist.csv")
        return self.model, (self.loss_hist)

    def infos(self):
        plt.figure(figsize=(15, 7))
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('MSE Loss')
        plt.plot(self.loss_hist)
        plt.savefig(self.run_dir + "/mse_loss.png")
        plt.show()
