import matplotlib.pyplot as plt
import numpy as np
from KDEpy import NaiveKDE

from downstream.downstream_base import DownstreamBase
from utils.common import batch


def fit_pykde(model, loader, bandwith=0, plot_density=False, **cfg):
    recon_err = []
    for data, _ in loader:
        data = data.float().to(model.device)
        recon = model(data)
        err = ((data - recon) ** 2).mean(axis=1).cpu().detach().numpy()
        recon_err += list(err.reshape(-1))
    pdf = NaiveKDE(bw=bandwith).fit(recon_err)
    if plot_density:
        plt.figure(figsize=(10, 6))
        plt.hist(recon_err, density=True, bins=100, alpha=0.5, label="Train data")
        xx = np.linspace(0, np.max(recon_err), 200)
        plt.plot(xx, pdf.evaluate(xx), label="KDE Fit")
        plt.legend()
        plt.show()
    return pdf


def get_pdf(type, model, loader, **cfg):
    if type == "kdepy":
        return fit_pykde(model, loader, **cfg)


class SingularRollingDensity(DownstreamBase):

    def __init__(self, model=None, batch_size=None, density={}, **cfg):
        super(SingularRollingDensity, self).__init__()
        self.loader = None
        self.model = model
        self.pdf = None
        self.density_cfg = density
        self.density_type = self.density_cfg["type"]
        self.density_cfg = self.density_cfg["config"]
        self.cfg = cfg
        self.batch_size = batch_size

    def fit(self, loader):
        self.loader = loader
        self.pdf = get_pdf(self.density_type, self.model, loader, **self.density_cfg)
        return self

    def predict(self, X):
        window_size = X.shape[1]
        batch_size = self.batch_size if self.batch_size is not None else X.shape[0]
        errs = []
        for b in batch(X, batch_size):
            recon = self.model(b)
            err = ((b - recon) ** 2).cpu().detach().numpy().mean(axis=1)
            errs.append(err)
        errs = np.concatenate(errs)
        probs = self.pdf.evaluate(errs)

        m = np.zeros((probs.shape[0], probs.shape[0] + window_size))
        for idx, rec in enumerate(probs):
            m[idx:idx + 1, idx:idx + window_size] = rec.repeat(window_size)
        means = np.array(np.ma.average(m, axis=0, weights=(m > 0))[:-window_size - 1])
        means[:window_size] = means[:window_size].mean().repeat(window_size)
        return means
