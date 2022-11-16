import torch
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, dtask, loader, dset, run_dir=None, **cfg):
    device = cfg["torch-device"]
    window_size = cfg["data"]["t_steps"]
    test_d = dset.test_data[0]
    y = dset.test_data[1]
    test_d = torch.Tensor(test_d).resize(test_d.shape[0], window_size, 1).to(device).float()
    probs = dtask.predict(test_d)
    found = y[probs.argmin()][0]
    print("found:", found)

    recs_test = model(test_d).detach().cpu().numpy()
    errs_test = np.mean((test_d.detach().cpu().numpy() - recs_test) ** 2, axis=(2, 1))
    m_test = np.zeros((errs_test.shape[0], errs_test.shape[0] + window_size))
    for idx, rec in enumerate(errs_test):
        m_test[idx:idx + 1, idx:idx + window_size] = rec
    means_test = np.array(np.ma.average(m_test, axis=0, weights=(m_test > 0))[:-window_size - 1])
    means_test[:window_size] = means_test[:window_size].mean().repeat(window_size)

    recs = model(torch.Tensor(dset.train_data[0]).to("cuda:0")).detach().cpu().numpy()
    errs = np.mean((dset.train_data[0]-recs)**2, axis=(2,1))
    m = np.zeros((errs.shape[0], errs.shape[0] + window_size))
    for idx, rec in enumerate(errs):
        m[idx:idx + 1, idx:idx + window_size] = rec
    means = np.array(np.ma.average(m, axis=0, weights=(m > 0))[:-window_size - 1])
    means[:window_size] = means[:window_size].mean().repeat(window_size)
    means_mal = means_test[y.reshape(-1)[:-1] == 1]
    means_norm = means_test[y.reshape(-1)[:-1] == 0]
    plt.hist(means, density=True, alpha=0.5, bins=100, label="train")
    plt.hist(means_mal, density=True, alpha=0.5, bins=2, label="test mal")
    plt.hist(means_norm, density=True, alpha=0.5, bins=100, label="test norm")
    plt.legend()
    plt.show()


    train_err = np.mean(((dset.train_data[0] - rec) ** 2), axis=(2, 1))
    mals_d = test_d[y.reshape(-1).nonzero()]
    norm_d = test_d[y.reshape(-1) == 0]
    norm_err = np.mean(((norm_d - model(norm_d)) ** 2).detach().cpu().numpy(), axis=(2, 1))
    err = np.mean(((mals_d - model(mals_d)) ** 2).detach().cpu().numpy(), axis=(2, 1))
    plt.hist(norm_err, density=True, bins=100, alpha=0.5, label="test normal")
    plt.hist(err, density=True, bins=20, alpha=0.5, label="test mal")
    plt.hist(train_err, density=True, bins=100, alpha=0.5, label="train norm")
    plt.legend()
    plt.show()

    mal_probs = probs[y.reshape(-1)[:-1] == 1]
    norm_probs = probs[y.reshape(-1)[:-1] == 0]

    plt.hist(mal_probs, density=True, alpha=0.5, label="mal probs", bins=20)
    plt.hist(norm_probs, density=True, alpha=0.5, label="norm_ probs", bins=20)
    plt.show()



    with open(run_dir + "/found.txt", "w+") as f:
        f.write(str(found) + "\n")
        f.write(str(probs.argmin))
