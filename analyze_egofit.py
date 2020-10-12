import argparse
from copy import deepcopy
from pathlib import Path
import warnings

import pickle
import pandas as pd
from libyana.exputils import argutils

parser = argparse.ArgumentParser()
parser.add_argument("--save_root", default="tmp")
parser.add_argument("--sort_loss", default="hand_v_dists")
args = parser.parse_args()
argutils.print_args(args)

save_root = Path(args.save_root)
results = []
df_data = []
for folder in save_root.iterdir():
    res_path = folder / "res.pkl"
    if res_path.exists():
        print(f"{res_path}")
        with open(res_path, "rb") as p_f:
            res = pickle.load(p_f)
            results.append(res)
        res_data = deepcopy(res["opts"])
        # Get last loss values
        for metric in res["losses"]:
            res_data[metric] = res["losses"][metric][-1]
        res_data[f"{metric}_l"] = len(res["losses"][metric])
        df_data.append(res_data)
    else:
        warnings.warn(f"Skipping missing {res_path}")
print(f"{res['losses'].keys()}")

df = pd.DataFrame(df_data)
df = df.sort_values(args.sort_loss)
print(df)
