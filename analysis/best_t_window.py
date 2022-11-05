from utils.common import fetch_remote_data, read_cfg
from pathlib import Path
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt


fetch_remote = False
resource = "/home/ubuntu/ts-anomaly-detection/out/runs/hp_sweep_window_size/generator_hp_sweep_window_size"
dst = "./data/window_size"
# d_loc = dst + "/generator_hp_sweep_window_size"
d_loc = dst + "/ts-anomaly-detection/out/runs/hp_sweep_window_size/generator_hp_sweep_window_size"

if fetch_remote:
    Path(dst).mkdir(parents=True, exist_ok=True)
    fetch_remote_data(resource, dst)


def construct_data_set(data_loc):
    columns = ["run-name", "set-nr", "window-size", "found", "location"]
    frame = pd.DataFrame(columns=columns, data=[])
    run_files = glob(data_loc + "/*")

    for file_path in run_files:
        if Path(file_path + "/found.txt").exists():
            run_name = file_path.split("/")[-1]
            try:
                cfg = read_cfg(file_path + "/run-config.yaml")
                found = 0.0
                with open(file_path + "/found.txt", "r") as f:
                    found = float(f.readline())
                set_nr = cfg["data"]["set_number"]
                t_steps = int(cfg["data"]["t_steps"])
                rec_data = [run_name, set_nr, t_steps, found, file_path]
                record = pd.DataFrame(columns=columns, data=[rec_data])
                frame = pd.concat([frame, record])
            except Exception as e:
                print(e)

    print(frame.head(10))
    frame_6 = frame[frame["set-nr"] == "006"]
    window_size_mean = frame_6.groupby(by="window-size")["found"].mean().reset_index()
    plt.plot(window_size_mean["window-size"], window_size_mean["found"])
    plt.scatter(window_size_mean["window-size"], window_size_mean["found"], marker="x")
    plt.show()


construct_data_set(d_loc)
