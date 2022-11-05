import os
import sys
import time
from itertools import product
from pathlib import Path

import yaml

from utils.common import replace_vars_in_cfg, ucr_set_to_window

if __name__ == '__main__':
    tmpl = sys.argv[1]
    tmpl_vals = sys.argv[2]

    n = 10
    generated_path = "./config/templates/augmentations-1/generated/"
    Path(generated_path).mkdir(parents=True, exist_ok=True)

    vals = yaml.load(open(tmpl_vals, "r"), Loader=yaml.FullLoader)

    cfg = yaml.load(open(tmpl, "r"), Loader=yaml.FullLoader)
    val_keys = vals.keys()

    for i in range(1, n + 1):
        combs = list(product(*vals.values()))
        for idx, comb in enumerate(combs):
            print("iter", i, "comb idx", idx, "of", len(combs))
            comb_dict = {key: comb[idx] for idx, key in enumerate(val_keys)}
            comb_dict["T_STEPS"] = ucr_set_to_window[comb_dict["SET_NUMBER"]]
            cfg_patched = replace_vars_in_cfg(cfg.copy(), comb_dict)
            str_comb = [str(v) for v in comb]
            f_name = str(hash(comb)) + ".yaml"
            file = generated_path + f_name
            with open(file, 'w+') as outfile:
                yaml.dump(cfg_patched, outfile, default_flow_style=False)
            t_start = time.time()
            os.system('python3 main.py ' + file)
            dur = time.time() - t_start
            print("seconds:", dur)
