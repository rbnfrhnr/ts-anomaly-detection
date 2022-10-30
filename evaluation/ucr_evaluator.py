import torch


def evaluate(model, dtask, loader, dset, run_dir=None, **cfg):
    device = cfg["torch-device"]
    window_size = cfg["data"]["t_steps"]
    test_d = dset.test_data[:, :-1]
    y = dset.test_data[:, -1]
    test_d = torch.Tensor(test_d).resize(test_d.shape[0], window_size, 1).to(device).float()
    probs = dtask.predict(test_d)
    found = y[probs.argmin()]
    print("found:", found)

    with open(run_dir + "/found.txt", "w+") as f:
        f.write(str(found) + "\n")
        f.write(str(probs.argmin))
