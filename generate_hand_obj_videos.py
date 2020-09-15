import argparse
import os
import pickle
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
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


parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument(
    "--epic_root",
    default="/sequoia/data2/yhasson/datasets/epic-kitchen/process_yana/frames_rgb_flow/rgb_frames/",
)
parser.add_argument("--interpolate", action="store_true")
parser.add_argument("--use_tar", action="store_true")
# parser.add_argument("--video_id", default=1, type=int)
parser.add_argument("--video_id", type=int)
# parser.add_argument("--person_id", default=1, type=int)
parser.add_argument("--person_id", type=int)
parser.add_argument("--verb_filter", type=str)
parser.add_argument("--frame_nb", default=100000, type=int)
parser.add_argument("--frame_step", default=10, type=int)
args = parser.parse_args()

if args.video_id is not None:
    args.video_id = f"{args.video_id:02d}"
if args.person_id is not None:
    args.person_id = f"P{args.person_id:02d}"

for key, val in vars(args).items():
    print(f"{key}: {val}")

annot_df = training_labels()
if args.video_id is not None:
    video_full_id = f"{args.person_id}_{args.video_id}"
else:
    video_full_id = None
if args.person_id is not None:
    annot_nb = len(annot_df)
    annot_df = annot_df[annot_df.participant_id == args.person_id]
    print(
        f"Kept {len(annot_df)} / {annot_nb} actions with participant id {args.person_id}"
    )
if args.video_id is not None:
    annot_nb = len(annot_df)
    annot_df = annot_df[annot_df.video_id == video_full_id]
    print(f"Kept {len(annot_df)} / {annot_nb} actions with video_id {video_full_id}")
if args.verb_filter is not None:
    annot_nb = len(annot_df)
    annot_df = annot_df[annot_df.verb == args.verb_filter]
    print(f"Kept {len(annot_df)} / {annot_nb} actions with verb {args.verb_filter}")
video_action_data = annot_df[(annot_df["video_id"] == video_full_id)]

# Get frame_idx : action_label frames
obj_df = labelutils.get_obj_labels(
    video_id=video_full_id, person_id=args.person_id, interpolate=args.interpolate
)
extended_action_labels, dense_df = labelutils.extend_action_labels(video_action_data)
action_names = set(extended_action_labels.values())

save_folder = Path("results/action_segm_videos")
if args.verb_filter is not None:
    save_folder = save_folder / args.verb_filter
os.makedirs(save_folder, exist_ok=True)
frame_template = os.path.join(args.epic_root, "{}/{}/{}/frame_{:010d}.jpg")

fig = plt.figure(figsize=(4, 5))
cmapping = displayutils.get_colors(action_names)

video_segm_idxs = dense_df.action_idx.unique()
resize_factor = 456 / 1920  # from original to target resolution
for video_segm_idx in video_segm_idxs:
    segm_images = []
    segm_df = dense_df[dense_df.action_idx == video_segm_idx]
    rendered_path = os.path.join(save_folder, f"{video_full_id}_{video_segm_idx}.webm")
    for frame_idx in tqdm(segm_df.frame_idx[:: args.frame_step]):
        frame_name = "frame_{frame_idx:010d}.jpg"
        frame_subpath = f"./{frame_name}"
        fig.clf()
        ax = fig.add_subplot(2, 1, 2)
        img_path = frame_template.format(
            args.split, args.person_id, video_full_id, frame_idx
        )
        if frame_idx in extended_action_labels:
            label = f"fr{frame_idx}: {extended_action_labels[frame_idx]}"
        else:
            label = f"fr{frame_idx}"
        if os.path.exists(img_path):
            displayutils.add_load_img(ax, img_path, label)
            boxes_df = obj_df[obj_df.frame == frame_idx]
            if boxes_df.shape[0] > 0:
                print("Box !")
                boxes = boxes_df.box.values
                labels = boxes_df.noun.values
                bboxes_norm = [
                    epic_box_to_norm(bbox, resize_factor=resize_factor)
                    for bbox in boxes
                ]
                label_color = "w"
                detect2d.visualize_bboxes(
                    ax, bboxes_norm, labels=labels, label_color=label_color
                )
        else:
            break
        # Get action label time extent bar for given frame
        adv_colors = labelutils.get_annot_adv(
            frame_idx, extended_action_labels, cmapping
        )
        ax = fig.add_subplot(2, 1, 1)
        ax.imshow(adv_colors)
        ax.axis("off")
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        segm_images.append(data)
    # score_clip = mpy.ImageSequenceClip(score_plots, fps=8)
    clip = mpy.ImageSequenceClip(segm_images, fps=8)
    # final_clip = mpy.clips_array([[clip,], [score_clip,]])
    clip.write_videofile(rendered_path)
    print(f"Saved video to {rendered_path}")
