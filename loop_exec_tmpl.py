import os
import re
import sys
import time
from itertools import product
from pathlib import Path

import yaml

path_matcher = re.compile(r'\$\{([^}^{]+)\}')


def path_constructor(loader, node, vals):
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    return os.environ.get(env_var) + value[match.end():]


def traverse(node, vals):
    if isinstance(node, dict):
        node = {k: traverse(v, vals) for k, v in node.items()}

    if isinstance(node, list):
        node = [traverse(v, vals) for v in node]

    if isinstance(node, str):
        match = path_matcher.match(node)
        if match:
            env_var = match.group()[2:-1]
            return vals[env_var]
        return node

    return node


if __name__ == '__main__':
    tmpl_vals = sys.argv[1]

    n = 5
    tmpl = "./config/templates/hp_vae_gen_tmpl.yaml"
    generated_path = "./config/templates/generated/"
    Path(generated_path).mkdir(parents=True, exist_ok=True)

    vals = yaml.load(open(tmpl_vals, "r"), Loader=yaml.FullLoader)

    cfg = yaml.load(open(tmpl, "r"), Loader=yaml.FullLoader)
    val_keys = vals.keys()
    sets = vals["SET_NUMBER"]

    for i in range(1, n + 1):
        combs = list(product(*vals.values()))
        for idx, comb in enumerate(combs):
            print("iter", i, "comb idx", idx, "of", len(combs))
            comb_dict = {key: comb[idx] for idx, key in enumerate(val_keys)}
            cfg_patched = traverse(cfg.copy(), comb_dict)
            str_comb = [str(v) for v in comb]
            f_name = "-".join(str_comb) + ".yaml"
            file = generated_path + f_name
            with open(file, 'w+') as outfile:
                yaml.dump(cfg_patched, outfile, default_flow_style=False)
            t_start = time.time()
            os.system('python3 main_model_only.py ' + file)
            dur = time.time() - t_start
            print("seconds:", dur)
