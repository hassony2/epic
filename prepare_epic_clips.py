import argparse
import os
import pickle
import traceback

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# Get FFMPEG outside env for moviepy

from epic_kitchens.meta import training_labels
from epic.hoa import handposes
from epic.egofit import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--epic_root", default="local_data/datasets/EPIC-KITCHENS")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--no_tar", action="store_true")
parser.add_argument("--no_filter", action="store_true")
parser.add_argument("--verbs", type=str, nargs="+", default=["open", "close"])
parser.add_argument("--nouns", type=str, nargs="+", default=["bottle"])
parser.add_argument("--frame_step", default=2, type=int)
parser.add_argument("--save_root", default="results/preprocess")
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

matplotlib.use("agg")
try:
    from epic.masks import bboxmasks
except Exception:
    traceback.print_exc()

labels = training_labels()
select_list = labels[(labels.noun == "bottle")]

# Initialize image parameters
resize_factor = 456 / 1920  # from original to target resolution
focal = 200
camintr = np.array([[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]])

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
mask_extractor = bboxmasks.MaskExtractor()

fig = plt.figure(figsize=(5, 4))
if not args.no_filter:
    annot_df = annot_df[
        annot_df.verb.isin(args.verbs) & annot_df.noun.isin(args.nouns)
    ]
print(
    f"Processing {len(annot_df)} samples with verbs {args.verbs} and nouns {args.nouns}"
)
# Remove new dataset
annot_df = annot_df[annot_df.video_id.str.len() == 6]

for annot_idx, annot in tqdm(annot_df.iterrows()):
    pickle_path = (
        f"{args.save_root}/"
        f"{annot.video_id}_{annot.start_frame}_{annot.stop_frame}_{args.frame_step}.pkl"
    )
    os.makedirs(args.save_root, exist_ok=True)
    if not os.path.exists(pickle_path):
        try:
            prepare.prepare_sequence(
                epic_root=args.epic_root,
                video_id=annot.video_id,
                start_frame=annot.start_frame,
                end_frame=annot.stop_frame,
                obj_path=args.obj_path,
                pickle_path=pickle_path,
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
                hoa_root=args.hoa_root,
            )
        except Exception:
            print(f"SKIPPING SEQUENCE {pickle_path}")
