import os
import random
from datetime import datetime
from pathlib import Path

import re
import numpy as np
import torch
import yaml
from scp import SCPClient
from paramiko import SSHClient, SFTPClient

ucr_sets = ['006', '022', '028', '033', '035', '053', '054', '059', '062', '070', '083', '102', '114',
            '119', '121', '123', '131', '138', '173', '193', '197', '221', '229', '236', '249']

ucr_set_to_window = {
    '006': 120,
    '022': 180,
    '028': 80,
    '033': 100,
    '035': 220,
    '053': 100,
    '054': 80,
    '059': 230,
    '062': 100,
    '070': 100,
    '083': 120,
    '102': 70,
    '114': 90,
    '119': 80,
    '121': 130,
    '123': 100,
    '131': 130,
    '138': 170,
    '173': 220,
    '193': 70,
    '197': 100,
    '221': 100,
    '229': 100,
    '236': 100,
    '249': 50
}

ctu_files = [
    'capture20110810.binetflow',
    'capture20110811.binetflow',
    'capture20110812.binetflow',
    'capture20110815.binetflow',
    'capture20110815-2.binetflow',
    'capture20110816.binetflow',
    'capture20110816-2.binetflow',
    'capture20110816-3.binetflow',
    'capture20110817.binetflow',
    'capture20110818.binetflow',
    'capture20110818-2.binetflow',
    'capture20110819.binetflow',
    'capture20110815-3.binetflow'
]

ctu_nr_to_name = {
    1: 'neris',
    2: 'neris',
    3: 'rbot',
    4: 'rbot',
    5: 'virut',
    6: 'menti',
    7: 'sogou',
    8: 'murlo',
    9: 'neris',
    10: 'rbot',
    11: 'rbot',
    12: 'nsis.ay',
    13: 'virut',
}


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


path_matcher = re.compile(r'\$\{([^}^{]+)\}')


def replace_vars_in_cfg(node, vals):
    if isinstance(node, dict):
        node = {k: replace_vars_in_cfg(v, vals) for k, v in node.items()}

    if isinstance(node, list):
        node = [replace_vars_in_cfg(v, vals) for v in node]

    if isinstance(node, str):
        match = path_matcher.match(node)
        if match:
            env_var = match.group()[2:-1]
            return vals[env_var]
        return node

    return node


def fetch_remote_data(src, dst):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect("160.85.252.139", username="ubuntu", key_filename="/home/robin/.ssh/zhaw-apu-frehnrob-test-2.pem")
    progress_fn = lambda file, size, sent: print("%s's progress: %.2f%%   \r" % (file, float(sent) / float(size) * 100))
    scp = SCPClient(ssh.get_transport(), progress=progress_fn, sanitize=lambda x: x)
    # scp.get(src + "/d**/*.yaml", recursive=True, local_path=dst)
    # scp.get(src + "/**/*.txt", recursive=True, local_path=dst)
    scp.get(src, recursive=True, local_path=dst)
