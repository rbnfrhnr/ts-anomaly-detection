from utils.common import read_cfg
import glob
from pathlib import Path
import torch
import pandas as pd
from data.ucr_loader import UCRDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import shutil

run_folder = "../out/runs/best_vae_generator/best_vae_generator/"
configs = glob.glob(run_folder + "**/run-config.yaml")
loss_fn = torch.nn.MSELoss(reduction="mean")
data = []
columns = ["set-nr", "t-steps", "mse", "model_file"]
dest_location = "./generators"

for cfg_file in configs:
    cfg = read_cfg(cfg_file)
    run_dir = "/".join(cfg_file.split("/")[:-1])
    set_nr = cfg["data"]["set_number"]
    t_steps = cfg["data"]["t_steps"]
    model_file = run_dir + "/model-checkpoints/model-final.pkl"
    if Path(model_file).exists():
        model = torch.load(model_file)
        best_model = torch.load(model_file)
        best_model.load_state_dict(torch.load(run_dir + "/model-checkpoints/model-best-state-dict.pkl"))
        train_hist = pd.read_csv(run_dir + "/train_hist.csv")
        min_val = train_hist[train_hist["loss"].min() == train_hist["loss"]]
        last = train_hist.iloc[-1]
        dataset = UCRDataset(**cfg["data"], cache_location=run_dir)
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=512,
                                             shuffle=True)
        mses = []
        for x, y in loader:
            x = x.to("cuda:0").float()
            rec = model(x)
            loss = loss_fn(x, rec)
            mses.append(loss.detach().cpu().numpy().tolist())

        data.append([set_nr, t_steps, np.mean(mses), model_file])
        print(mses)

        # xx = torch.Tensor(dataset.train_data[0][100])
        # xx = xx.reshape((1, *xx.shape)).to("cuda:0")
        # z = model.encoder(xx)
        # zz = best_model.encoder(xx)
        # new_z = z + torch.normal(torch.zeros_like(z), torch.zeros_like(z) + 0.05)
        # new_zz = z + torch.normal(torch.zeros_like(zz), torch.zeros_like(zz) + 0.05)
        # new_x = model.decoder(new_z)
        # new_xx = best_model.decoder(new_zz)
        #
        # plt.plot(xx.detach().cpu().reshape(xx.shape[1]))
        # plt.plot(new_x.detach().cpu().reshape(xx.shape[1]))
        # plt.plot(new_xx.detach().cpu().reshape(xx.shape[1]))
        # plt.plot(savgol_filter(new_xx.detach().cpu().reshape(xx.shape[1]), 40, 5))
        #
        # plt.show()

frame = pd.DataFrame(columns=columns, data=data)
best_generators = frame[frame["mse"].isin(frame.groupby(by=["set-nr"])["mse"].min())]
Path(dest_location).mkdir(parents=True, exist_ok=True)

for idx, rec in best_generators.iterrows():
    dest_file = dest_location + "/" + rec["set-nr"] + "-" + str(rec["t-steps"]) + "-generator.pkl"
    shutil.copy(rec["model_file"], dest_file)

print("here")
