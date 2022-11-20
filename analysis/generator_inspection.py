import torch
import matplotlib.pyplot as plt
from utils.common import ucr_set_to_window
from data.ucr_loader import UCRDataset
import math
from scipy.signal import savgol_filter

# good: 022, 035, (114) (173)
smooth = ["062", "022", "035", "102", "114", "173", "193"]
# good: 121
no_smooth = ["006", "028", "121"]
same = ["033", "054", "070", "119", "138"]
# good 069
neither = ["053", "059", "229", "249"]
data_location = "/home/robin/Documents/lbnl/crd/datasets/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"
generator_path = './generators'
device = "cuda:0"



def create_data_cfg(set_nr):
    data_cfg = {"set_number": set_nr, "location": data_location, "data-set": "ucr",
                "t_steps": ucr_set_to_window[set_nr]}
    return data_cfg

ucr_sets = ['006', '022', '028', '033', '035', '053', '054', '059', '062', '070',
            '083', '102', '114', '119', '121', '123', '131', '138', '173', '193',
            '197', '221', '229', '236', '249']


ok = ["022", "033", "070", "121", "138", "173"]
meh = ["006", "028", "035", "102", "114", "119", "131","193", "197", "236"]
nope = ["053", "054", "059", "062", "083", "123", "221", "229", "249"]

cfg = create_data_cfg("197")
generator_path = generator_path + "/" + cfg["set_number"] + "-" + str(cfg["t_steps"]) + "-generator.pkl"
generator = torch.load(generator_path)
dataset = UCRDataset(**cfg)
sample = dataset[650][0].reshape(1, *dataset[0][0].shape)
z = generator.encoder(torch.Tensor(sample).to(device))
means = torch.zeros((sample.shape[0], 4)).to(device)
stdv = torch.zeros((sample.shape[0], 4)).to(device) + 0.1
new_z = z + torch.normal(mean=means, std=stdv)
new_x = generator.decoder(new_z)
new_x = new_x.detach().cpu().numpy()
filter_window = min(40, math.floor((new_x.shape[1] * 0.5)))
polyorder = min(filter_window - 1, 7)
smoothed = savgol_filter(new_x.reshape(-1, cfg["t_steps"]), filter_window, polyorder)
print(len(dataset))
plt.plot(sample.reshape(-1), label="original")
plt.plot(new_x.reshape(-1), label="non-smoothed", alpha=0.5)
plt.plot(smoothed.reshape(-1), label="smoothed", alpha=0.5)
plt.legend()
plt.show()
