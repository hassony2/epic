import argparse
import os
from pathlib import Path
import pickle
import traceback

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
import torch

# Get FFMPEG outside env for moviepy
import os

os.environ["FFMPEG_BINARY"] = "/sequoia/data3/yhasson/miniconda3/bin/ffmpeg"
import moviepy.editor as mpy

from tqdm import tqdm

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d
from libyana.metrics.iou import get_iou
from libyana.transformutils.handutils import get_affine_transform, transform_img

from epic.boxutils import (
    epic_box_to_norm,
    get_center_scale,
    get_closest,
    extend_props,
    extend_labels,
)
from epic import displayutils
from epic import labelutils
from epic.viz import hoaviz, boxgtviz, masksviz
from epic.hoa import gethoa
from epic.masks import grabmasks
from epic.fitting import fitobj

try:
    from epic.masks import bboxmasks
except Exception:
    traceback.print_exc()

try:
    from epic.masks import getmasks
except Exception:
    traceback.print_exc()
from epic.hpose import handposes, handviz


parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--show_adv", action="store_true")
parser.add_argument(
    "--epic_root",
    default="/sequoia/data2/yhasson/datasets/epic-kitchen/process_yana/frames_rgb_flow/rgb_frames/",
)
parser.add_argument("--fps", default=2, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--video_id", type=str, default="P02_09")
parser.add_argument("--start_frame", type=int, default=104309)
parser.add_argument("--end_frame", type=int, default=104474)
parser.add_argument("--gt_objects", action="store_true")
parser.add_argument("--frame_nb", default=100000, type=int)
parser.add_argument("--frame_step", default=10, type=int)
parser.add_argument("--faces_per_pixel", default=10, type=int)
parser.add_argument("--pickle_path")
parser.add_argument(
    "--mask_mode", default="grabcut", help=["epic", "maskrcnn", "grabcut"]
)
parser.add_argument("--hands", action="store_true", help="Add predicted hand poses")
parser.add_argument(
    "--hoa_root",
    default="/sequoia/data2/dataset/epic-100/3l8eci2oqgst92n14w2yqi5ytu/hand-objects/",
)
parser.add_argument(
    "--obj_path",
    default="/sequoia/data2/dataset/shapenet/ShapeNetCore.v2/02880940/95ac294f47fd7d87e0b49f27ced29e3/models/model_normalized.obj",
)
args = parser.parse_args()

# Load hand object detections
hoa_dets = gethoa.load_video_hoa(args.video_id, hoa_root=args.hoa_root)
resize_factor = 456 / 1920  # from original to target resolution
annot_df = training_labels()
video_df = annot_df[annot_df.video_id == args.video_id]
# extended_action_labels, dense_df = labelutils.extend_action_labels(video_df)
mask_extractor = bboxmasks.MaskExtractor()

frame_template = os.path.join(args.epic_root, "{}/{}/{}/frame_{:010d}.jpg")
fig = plt.figure(figsize=(5, 4))
dump_list = []

for frame_idx in tqdm(range(args.start_frame, args.end_frame, args.frame_step)):
    frame_name = "frame_{frame_idx:010d}.jpg"
    frame_subpath = f"./{frame_name}"
    fig.clf()
    if args.show_adv:
        ax = fig.add_subplot(2, 1, 2)
    else:
        ax = fig.add_subplot(1, 1, 1)
    img_path = frame_template.format(
        args.split, args.video_id[:3], args.video_id, frame_idx
    )
    label = f"fr{frame_idx}"
    img = Image.open(img_path)
    img = np.array(img)
    ax.imshow(img)
    hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
    with torch.no_grad():
        res = mask_extractor.masks_from_df(img, hoa_df, resize_factor=resize_factor)
    labels = [
        f"{cls}: {score:.2f}" for cls, score in zip(res["classes"], res["scores"])
    ]
    masksviz.add_masks_viz(
        ax,
        res["masks"],
        res["boxes"],
        labels=labels,
        debug=args.debug,
    )
    fig.savefig("tmp.png")
    mask = res["masks"][0]
    if args.pickle_path is not None:
        dump_list.append({"mask": mask, "obj_path": args.obj_path})
if args.pickle_path is not None:
    with open(args.pickle_path, "wb") as p_f:
        pickle.dump(dump_list, p_f)
    print(f"Saved info to {args.pickle_path}")
# fitres = fitobj.fitobj2mask(mask, args.obj_path)
import pdb

pdb.set_trace()
