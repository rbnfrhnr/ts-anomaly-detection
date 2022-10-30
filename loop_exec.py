import os

import yaml

from utils.common import read_cfg

if __name__ == '__main__':
    n = 5
    ae_cfg_file = "./config/ae_test.yaml"
    vae_cfg_file = "./config/vae_test.yaml"
    generated_path = "./config/generated/"
    sets = [59, 114, 173, 6, 121, 193, 53, 35, 197, 70, 119, 22, 221,
            33, 62, 102, 123, 83, 249, 54, 236, 138, 229, 28]
    ae_cfg = read_cfg(ae_cfg_file)
    # vae_cfg = read_cfg(vae_cfg_file)

    for i in range(1, n + 1):
        for subset in sets:
            ae_cfg["data"]["set_number"] = '0' + str(subset) if subset < 100 else str(subset)
            ae_cfg["data"]["set_number"] = '0' + ae_cfg["data"]["set_number"] if subset < 10 else ae_cfg["data"]["set_number"]

            # vae_cfg["data"]["set_number"] = '0' + str(subset) if subset < 100 else str(subset)
            ae_name = "ae-" + str(subset) + ".yaml"
            # vae_name = "vae-" + str(subset) + ".yaml"
            ae_file = generated_path + ae_name
            # vae_file = generated_path + vae_name
            with open(ae_file, 'w+') as outfile:
                yaml.dump(ae_cfg, outfile, default_flow_style=False)
            # with open(vae_file, 'w+') as outfile:
            #     yaml.dump(vae_cfg, outfile, default_flow_style=False)

            print("ae", subset)
            os.system('python3 main.py ' + ae_file)
            # print("vae", subset)
            # os.system('python3 main.py ' + vae_file)
