from utils.common import read_cfg
import glob
from pathlib import Path
import torch
import pandas as pd
from data.ucr_loader import UCRDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# run_folder = "./data/window_size/ts-anomaly-detection/out/runs/hp_sweep_window_size/generator_hp_sweep_window_size/"
run_folder = "./data/augmentations-exp-1/"
# run_folder = "../out/runs/best_vae_generator/best_vae_generator/"
configs = glob.glob(run_folder + "**/**/run-config.yaml")

# frame = pd.DataFrame(columns=columns, data[])
data = []
columns=["model_type", "augmentation","set-nr", "found", "location"]

for cfg_file in configs:
    try:
        cfg = read_cfg(cfg_file)
        run_dir = "/".join(cfg_file.split("/")[:-1])
        found_file = run_dir + "/found.txt"
        if Path(found_file).exists():
            model_type = cfg["autoencoder"]["type"]
            augmentation = cfg["augmentation"]["name"]
            found = 0.0
            set_nr = cfg["data"]["set_number"]
            with open(found_file, "r") as f:
                found = float(f.readline())
            data.append([model_type, augmentation, set_nr,found, run_dir])
    except Exception as e:
        print("could not process " + cfg_file)
frame = pd.DataFrame(columns=columns, data=data)

perf = frame.groupby(by=["model_type", "set-nr","augmentation"])["found"].agg(mean_accuracy=np.mean, count=lambda x: np.sum(0 <= x)).reset_index()
print(frame.head(10))

