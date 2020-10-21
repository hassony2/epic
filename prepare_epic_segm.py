import argparse
import pickle
import traceback

import matplotlib

from matplotlib import pyplot as plt

import numpy as np

# Get FFMPEG outside env for moviepy

from epic_kitchens.meta import training_labels
from epic.hoa import gethoa, handposes
from epic.egofit import prepare

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

# Initialize image parameters
resize_factor = 456 / 1920  # from original to target resolution
focal = 200
camintr = np.array([[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]])
hoa_dets = gethoa.load_video_hoa(args.video_id, hoa_root=args.hoa_root)

# Load hand pose estimator
hand_checkpoint = (
    "assets/handmocap/extra_data/hand_module/"
    "pretrained_weights/pose_shape_best.pth"
)
smpl_folder = "assets/handmocap/extra_data/smpl"
hand_extractor = handposes.HandExtractor(hand_checkpoint, smpl_folder)

# Epic-55 annotations
annot_df = training_labels()
# Epic-100 annotations
with open(f"assets/EPIC_100_{args.split}.pkl", "rb") as p_f:
    annot_df = pickle.load(p_f)
video_df = annot_df[annot_df.video_id == args.video_id]
# extended_action_labels, dense_df = labelutils.extend_action_labels(video_df)
mask_extractor = bboxmasks.MaskExtractor()

fig = plt.figure(figsize=(5, 4))
prepare.prepare_sequence(
    args.epic_root,
    args.video_id,
    args.start_frame,
    args.end_frame,
    args.obj_path,
    pickle_path=args.pickle_path,
    frame_step=args.frame_step,
    fig=fig,
    show_adv=False,
    split=args.split,
    no_tar=args.no_tar,
    resize_factor=resize_factor,  # from original to target resolution
    mask_extractor=mask_extractor,
    hand_extractor=hand_extractor,
    debug=args.debug,
    camintr=camintr,
    hoa_dets=hoa_dets,
    hoa_root=args.hoa_root,
)
