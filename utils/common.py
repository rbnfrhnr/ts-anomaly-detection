import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ucr_sets = [6, 22, 28, 33, 35, 53, 54, 59, 62, 70, 83, 102, 114, 119, 121,
            123, 131, 138, 173, 193, 197, 221, 229, 236, 249]


def read_cfg(cfg_file):
    with open(cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    if cfg is None:
        print('no cfg present')
    return cfg


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def set_seeds(cfg):
    # Seed value
    default_seed = int(round(999999 * random.random()) + 1)
    seed_value = cfg['global-seed'] if 'global-seed' in cfg else None
    seed_value = seed_value if seed_value is not None else default_seed
    cfg['global-seed'] = seed_value

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set torch manual seed
    torch.manual_seed(seed_value)


def setup_run(logging={}, **cfg):
    d = datetime.today()
    log_location = logging['log-location']
    logger_name = logging['logger-name']
    cfg['run-id'] = cfg['experiment-name'] + '-' + d.strftime("%Y-%m-%d-%H-%M-%S")
    cfg['run-dir'] = log_location + "/" + cfg['experiment-name'] + '/' + cfg['run-id']
    run_dir = cfg['run-dir']
    Path(cfg['run-dir']).mkdir(parents=True, exist_ok=True)
    set_seeds(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device if "use_gpu" in cfg and cfg["use_gpu"] else "cpu"
    cfg["torch-device"] = device
    with open(r'' + run_dir + '/run-config.yaml', 'w') as file:
        documents = yaml.dump(cfg, file)
    return cfg
