import argparse
import os
import traceback

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
import torch

# Get FFMPEG outside env for moviepy
from tqdm import tqdm

from epic.viz import masksviz, handviz, hoaviz
from epic.hoa import gethoa, handposes
from epic.io.tarutils import TarReader
from epic.tracking import trackhoadf

matplotlib.use("agg")
try:
    from epic.masks import bboxmasks
except Exception:
    traceback.print_exc()

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--show_adv", action="store_true")
parser.add_argument("--epic_root", default="local_data/datasets/EPIC-KITCHENS")
parser.add_argument("--fps", default=2, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--no_tar", action="store_true")
parser.add_argument("--video_id", type=str, default="P01_01")
parser.add_argument("--start_frame", type=int, default=1)
parser.add_argument("--end_frame", type=int)
parser.add_argument("--gt_objects", action="store_true")
parser.add_argument("--frame_nb", default=100000, type=int)
parser.add_argument("--frame_step", default=10, type=int)
parser.add_argument("--faces_per_pixel", default=10, type=int)
parser.add_argument("--pickle_path")
parser.add_argument(
    "--mask_mode", default="grabcut", help=["epic", "maskrcnn", "grabcut"]
)
parser.add_argument(
    "--hands", action="store_true", help="Add predicted hand poses"
)
parser.add_argument("--hoa_root", default="local_data/datasets/epic-hoa")
parser.add_argument(
    "--obj_path",
    default=(
        # "/sequoia/data2/dataset/shapenet/ShapeNetCore.v2/"
        # Plate
        # "02880940/95ac294f47fd7d87e0b49f27ced29e3/models/model_normalized.obj"
        "/gpfsstore/rech/tan/usk19gv/datasets/shapenet/"
        # Bottle
        "02876657/d851cbc873de1c4d3b6eb309177a6753/models/model_normalized.obj"
    ),
)
args = parser.parse_args()

if not args.no_tar:
    tareader = TarReader()

# Prepare models
hand_checkpoint = (
    "assets/handmocap/extra_data/hand_module/"
    "pretrained_weights/pose_shape_best.pth"
)
smpl_folder = "assets/handmocap/extra_data/smpl"
mask_extractor = bboxmasks.MaskExtractor()
hand_extractor = handposes.HandExtractor(hand_checkpoint, smpl_folder)

# Initialize image parameters
resize_factor = 456 / 1920  # from original to target resolution
focal = 200
camintr = np.array([[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]])
hoa_dets = gethoa.load_video_hoa(args.video_id, hoa_root=args.hoa_root)
dt = 0.02

info_df = None
fig = plt.figure()
if args.end_frame is None:
    args.end_frame = max(hoa_dets.frame)
tracked_dets = trackhoadf.track_hoa_df(
    hoa_dets,
    video_id=args.video_id,
    start_frame=args.start_frame,
    end_frame=args.end_frame,
    dt=dt,
    object_only=True,
)
tracks = (tracked_dets.groupby("track_id").frame.nunique() > 30).sum()
track_lengths = tracked_dets.groupby("track_id").frame.nunique()
min_track_size = 10
print(
    f"Got {len(track_lengths)} tracks for {args.end_frame}"
    f"out of which {len(track_lengths[track_lengths > min_track_size])}"
    f"of length > {min_track_size}"
)

frame_template = os.path.join(args.epic_root, "{}/{}/{}/frame_{:010d}.jpg")

for frame_idx in tqdm(
    range(args.start_frame, args.end_frame, args.frame_step)
):
    frame_name = "frame_{frame_idx:010d}.jpg"
    frame_subpath = f"./{frame_name}"
    fig.clf()
    img_path = frame_template.format(
        args.split, args.video_id[:3], args.video_id, frame_idx
    )
    if args.no_tar:
        img = Image.open(img_path)
        img = np.array(img)
    else:
        img = tareader.read_tar_frame(img_path)
        img = img[:, :, ::-1]
    print(img_path)
    label = f"fr{frame_idx}"
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.set_title("untracked results")
    hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
    with torch.no_grad():
        # Extract object masks
        res = mask_extractor.masks_from_df(
            img, hoa_df, resize_factor=resize_factor
        )
        # Extract hands
        pred_hands = hand_extractor.hands_from_df(
            img, hoa_df, resize_factor=resize_factor
        )

        if len(pred_hands):
            # Draw hand renderings
            handviz.add_hands_viz(ax, img, pred_hands, camintr)

            # Draw hand boxes
            hoaviz.add_hoa_viz(
                ax, hoa_df, resize_factor=resize_factor, debug=args.debug
            )
        if len(res["masks"]):  # TODO remove
            labels = [
                f"{cls}: {score:.2f}"
                for cls, score in zip(res["classes"], res["scores"])
            ]
            masksviz.add_masks_viz(
                ax, res["masks"], res["boxes"], labels=labels, debug=args.debug
            )
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    ax.set_title("tracked results")
    hoa_df = tracked_dets[tracked_dets.frame == frame_idx]
    with torch.no_grad():
        # Extract object masks
        res = mask_extractor.masks_from_df(
            img, hoa_df, resize_factor=resize_factor
        )
        # Extract hands
        pred_hands = hand_extractor.hands_from_df(
            img, hoa_df, resize_factor=resize_factor
        )

        if len(pred_hands):
            # Draw hand renderings
            handviz.add_hands_viz(ax, img, pred_hands, camintr)

            # Draw hand boxes
            hoaviz.add_hoa_viz(
                ax, hoa_df, resize_factor=resize_factor, debug=args.debug
            )
        if len(res["masks"]):  # TODO remove
            labels = [
                f"{cls}: {score:.2f}"
                for cls, score in zip(res["classes"], res["scores"])
            ]
            masksviz.add_masks_viz(
                ax, res["masks"], res["boxes"], labels=labels, debug=args.debug
            )
    fig.savefig(f"tmp/imgs/tmp_{frame_idx:05d}.png")
