import argparse
import os
import pickle
import traceback

import matplotlib

from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
import torch

# Get FFMPEG outside env for moviepy
from tqdm import tqdm

from epic_kitchens.meta import training_labels
from epic.viz import masksviz, handviz, hoaviz
from epic.hoa import gethoa, handposes
from epic.io.tarutils import TarReader

matplotlib.use("agg")
try:
    from epic.masks import bboxmasks
except Exception:
    traceback.print_exc()

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--show_adv", action="store_true")
parser.add_argument(
    "--epic_root", default="local_data/datasets/EPIC-KITCHENS",
)
parser.add_argument("--fps", default=2, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--no_tar", action="store_true")
parser.add_argument("--video_id", type=str, default="P01_01")
parser.add_argument("--start_frame", type=int, default=1)
parser.add_argument("--end_frame", type=int, default=104)
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
parser.add_argument(
    "--hoa_root",
    default="local_data/datasets/epic-hoa",
)
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

# Initialize image parameters
resize_factor = 456 / 1920  # from original to target resolution
focal = 200
camintr = np.array([[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]])
hoa_dets = gethoa.load_video_hoa(args.video_id, hoa_root=args.hoa_root)

# Load hand pose estimator
hand_checkpoint = ("assets/handmocap/extra_data/hand_module/"
                   "pretrained_weights/pose_shape_best.pth")
smpl_folder = "assets/handmocap/extra_data/smpl"
hand_extractor = handposes.HandExtractor(hand_checkpoint, smpl_folder)

# Epic-55 annotations
# annot_df = training_labels()
# Epic-100 annotations
with open(f"assets/EPIC_100_{args.split}.pkl", "rb") as p_f:
    annot_df = pickle.load(p_f)
video_df = annot_df[annot_df.video_id == args.video_id]
# extended_action_labels, dense_df = labelutils.extend_action_labels(video_df)
mask_extractor = bboxmasks.MaskExtractor()

frame_template = os.path.join(args.epic_root, "{}/{}/{}/frame_{:010d}.jpg")
fig = plt.figure(figsize=(5, 4))
dump_list = []


for frame_idx in tqdm(
    range(args.start_frame, args.end_frame, args.frame_step)
):
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
    if args.no_tar:
        img = Image.open(img_path)
        img = np.array(img)
    else:
        img = tareader.read_tar_frame(img_path)
        img = img[:, :, ::-1]
    print(img_path)
    label = f"fr{frame_idx}"
    ax.imshow(img)
    hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
    with torch.no_grad():
        # Extract object masks
        res = mask_extractor.masks_from_df(
            img, hoa_df, resize_factor=resize_factor
        )
        # Extract hands
        pred_hands = hand_extractor.hands_from_df(img, hoa_df, resize_factor=resize_factor)

        if len(pred_hands):
            # Draw hand renderings
            handviz.add_hands_viz(ax, img, pred_hands, camintr)

            # Draw hand boxes
            hoaviz.add_hoa_viz(
                ax, hoa_df, resize_factor=resize_factor, debug=args.debug
            )
        if len(res["masks"]): # TODO remove
            labels = [
                f"{cls}: {score:.2f}"
                for cls, score in zip(res["classes"], res["scores"])
            ]
            masksviz.add_masks_viz(
                ax, res["masks"], res["boxes"], labels=labels, debug=args.debug
            )
            mask = res["masks"][0]
            boxes = res["boxes"][0]
        else:
            mask = None
            boxes = None
        if args.debug:
            fig.savefig(f"tmp_{frame_idx:05d}.png")
        print(boxes)
        if args.pickle_path is not None:
            dump_list.append(
                {"mask": mask, "boxes": boxes, "obj_path": args.obj_path, "hands": pred_hands}
            )
if args.pickle_path is not None:
    with open(args.pickle_path, "wb") as p_f:
        pickle.dump(dump_list, p_f)
    print(f"Saved info to {args.pickle_path}")
