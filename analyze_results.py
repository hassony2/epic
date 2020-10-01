import argparse
from copy import deepcopy
import os
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
import torch
import numpy as np
import pickle
import pandas as pd
from epic.lib3d import rotations

parser = argparse.ArgumentParser()
parser.add_argument("--save_root", default="tmp")
args = parser.parse_args()

save_root = Path(args.save_root)
results = []
df_data = []
for folder in save_root.iterdir():
    res_path = folder / "res.pkl"
    if res_path.exists():
        with open(res_path, "rb") as p_f:
            res = pickle.load(p_f)
            results.append(res)
        res_data = deepcopy(res["opts"])
        res_data.pop("obj_paths")
        for metric in ["l1", "l2", "mask"]:
            res_data[metric] = res["metrics"][metric][-1]
        res_data["metric_l"] = len(res["metrics"][metric])
        df_data.append(res_data)
    else:
        warnings.warn(f"Skipping missing {res_path}")

df = pd.DataFrame(df_data)
ref_metric = "mask"
df = df.sort_values(ref_metric)
results = sorted(results, key=lambda res: res["metrics"][ref_metric][-1])
strs = [
    f"{res['opts']['loss_type']} lr {res['opts']['lr']} {res['metrics'][ref_metric][-1]:.2f}"
    for res in results
]
print(df)
import pdb

pdb.set_trace()
