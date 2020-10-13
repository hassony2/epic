import argparse
from pathlib import Path
import shutil
from copy import deepcopy
import warnings

import pickle
import pandas as pd

from epic.egofit import logutils
from libyana.exputils import argutils
from moviepy import editor


def make_gif(img_paths, gif_path, fps=2):
    img_paths = [str(path) for path in img_paths]
    clip = editor.ImageSequenceClip(img_paths, fps=fps)
    clip.write_gif(gif_path)


def make_video(img_paths, video_path, fps=2):
    img_paths = [str(path) for path in img_paths]
    clip = editor.ImageSequenceClip(img_paths, fps=fps)
    clip.write_videofile(str(video_path))


parser = argparse.ArgumentParser()
parser.add_argument("--save_root", default="tmp")
parser.add_argument("--sort_loss", default="hand_v_dists")
parser.add_argument("--destination", default="results/tables")
parser.add_argument("--gifs", action="store_true")
parser.add_argument("--no_videos", action="store_true")

args = parser.parse_args()
argutils.print_args(args)

destination = Path(args.destination)
destination.mkdir(exist_ok=True, parents=True)

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

        # Get last optimized image
        img_paths = res["imgs"]
        img_path = img_paths[list(img_paths)[-1]]
        res_data["last_img_path"] = img_path
        # Generate gif
        if args.gifs:
            gif_path = folder / "optim.gif"
            make_gif(img_paths.values(), gif_path)
            res_data["optim_img_path"] = str(gif_path)
        if not args.no_videos:
            video_path = folder / "optim.webm"
            make_video(img_paths.values(), video_path)
            res_data["optim_video_path"] = str(video_path)
        df_data.append(res_data)
    else:
        warnings.warn(f"Skipping missing {res_path}")
print(f"{res['losses'].keys()}")

df = pd.DataFrame(df_data)
df = df.sort_values(args.sort_loss)
print(df)
df_html = logutils.df2html(df, local_folder=destination)
with (destination / "raw.html").open("w") as h_f:
    h_f.write(df_html)
shutil.copy("htmlassets/index.html", destination / "index.html")
