import argparse
import os
from pathlib import Path
import traceback

import matplotlib
from matplotlib import pyplot as plt
import moviepy.editor as mpy
import numpy as np
import pandas as pd
from PIL import Image
import torch
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
from epic.hpose import handviz

matplotlib.use("agg")
try:
    from epic.masks import bboxmasks
except Exception:
    traceback.print_exc()

try:
    from epic.masks import getmasks
except Exception:
    traceback.print_exc()

try:
    from epic.hpose import handposes
except Exception:
    traceback.print_exc()


parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--show_adv", action="store_true")
parser.add_argument(
    "--epic_root",
    default="local_data/datasets/EPIC-KITCHENS",
)
parser.add_argument("--use_tar", action="store_true")
parser.add_argument("--fps", default=2, type=int)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--video_ids", type=int, nargs="+")
parser.add_argument("--person_ids", type=int, nargs="+")
parser.add_argument("--verb_filter", type=str)
parser.add_argument("--noun_filters", type=str, nargs="+")
parser.add_argument("--gt_objects", action="store_true")
parser.add_argument("--frame_nb", default=100000, type=int)
parser.add_argument("--frame_step", default=10, type=int)
parser.add_argument(
    "--mask_mode", default="grabcut", help=["epic", "maskrcnn", "grabcut"]
)
parser.add_argument(
    "--hoa", action="store_true", help="Add predicted hand and object bbox annotations"
)
parser.add_argument("--hands", action="store_true", help="Add predicted hand poses")
parser.add_argument(
    "--hoa_root",
    default="local_data/datasets/epic-hoa",
)
args = parser.parse_args()

if args.video_ids is not None:
    args.video_ids = [f"{video_id:02d}" for video_id in args.video_ids]
    if args.person_ids is None:
        raise ValueError("--person_ids should be provided when --video_ids is provided")

if args.person_ids is not None:
    args.person_ids = [f"P{person_id:02d}" for person_id in args.person_ids]

for key, val in vars(args).items():
    print(f"{key}: {val}")

annot_df = training_labels()
if args.video_ids is not None:
    video_full_ids = []
    for person_id, video_id in zip(args.person_ids, args.video_ids):
        video_full_ids.append(f"{person_id}_{video_id}")
    annot_nb = len(annot_df)
    annot_df = annot_df[annot_df.video_id.isin(video_full_ids)]
    print(f"Kept {len(annot_df)} / {annot_nb} actions with video_id {video_full_ids}")
else:
    video_full_ids = annot_df.video_id.unique()
    args.person_ids = [vid[:3] for vid in video_full_ids]
    args.video_ids = [vid[-2:] for vid in video_full_ids]
if args.verb_filter is not None:
    annot_nb = len(annot_df)
    annot_df = annot_df[annot_df.verb == args.verb_filter]
    print(f"Kept {len(annot_df)} / {annot_nb} actions with verb {args.verb_filter}")
if args.noun_filters is not None:
    annot_nb = len(annot_df)
    annot_df = annot_df[annot_df.noun.isin(args.noun_filters)]
    print(f"Kept {len(annot_df)} / {annot_nb} actions with nouns {args.noun_filters}")

# Get frame_idx : action_label frames
obj_dfs = []
for video_full_id in video_full_ids:
    obj_df = labelutils.get_obj_labels(
        video_id=video_full_id, person_id=video_full_id[:3], interpolate=True
    )
    obj_dfs.append(obj_df)
obj_df = pd.concat(obj_dfs)

extended_action_labels, dense_df = labelutils.extend_action_labels(annot_df)
action_names = set(extended_action_labels.values())

save_folder = Path(f"results/action_segms_{args.mask_mode}/")
if args.verb_filter is not None:
    save_folder = save_folder / args.verb_filter
if args.noun_filters is not None:
    save_folder = save_folder / "_".join(args.noun_filters)
os.makedirs(save_folder, exist_ok=True)
frame_template = os.path.join(args.epic_root, "{}/{}/{}/frame_{:010d}.jpg")

if args.show_adv:
    fig = plt.figure(figsize=(4, 5))
else:
    fig = plt.figure(figsize=(5, 4))
cmapping = displayutils.get_colors(action_names)

video_segm_idxs = dense_df.action_idx.unique()
resize_factor = 456 / 1920  # from original to target resolution
for video_segm_idx in video_segm_idxs:
    try:
        segm_images = []
        segm_df = dense_df[dense_df.action_idx == video_segm_idx]
        video_full_id = segm_df.video_id.values[0]
        if args.hoa:
            hoa_dets = gethoa.load_video_hoa(video_full_id, hoa_root=args.hoa_root)
            if args.mask_mode == "epic":
                masks_dets = getmasks.load_video_masks(
                    video_full_id,
                    masks_root=args.hoa_root.replace("hand-objects", "masks"),
                    hoa_df=hoa_dets,
                    filter_mode="hoaiou",
                )
            elif args.mask_mode == "maskrcnn":
                mask_extractor = bboxmasks.MaskExtractor()
            elif args.mask_mode == "grabcut":
                pass
            else:
                raise ValueError(f"mask_mode {args.mask_mode} not in [epic|maskrcnn]")
        if args.gt_objects:
            obj_df = labelutils.get_obj_labels(
                video_id=video_full_id, person_id=video_full_id[:3], interpolate=True
            )
        person_id = segm_df.participant_id.values[0]
        verb = segm_df.verb.values[0]
        noun = segm_df.noun.values[0]
        rendered_path = os.path.join(
            save_folder,
            f"{video_full_id}_{video_segm_idx}_verb_{verb}_noun_{noun}.webm",
        )
        for frame_idx in tqdm(segm_df.frame_idx[:: args.frame_step]):
            frame_name = "frame_{frame_idx:010d}.jpg"
            frame_subpath = f"./{frame_name}"
            fig.clf()
            if args.show_adv:
                ax = fig.add_subplot(2, 1, 2)
            else:
                ax = fig.add_subplot(1, 1, 1)
            img_path = frame_template.format(
                args.split, person_id, video_full_id, frame_idx
            )
            if frame_idx in extended_action_labels:
                label = f"fr{frame_idx}: {extended_action_labels[frame_idx]}"
            else:
                label = f"fr{frame_idx}"
            if os.path.exists(img_path):
                img = displayutils.add_load_img(ax, img_path, label)

                # Display object annotations (ground truth)
                vid_df = obj_df[obj_df.video_id == video_full_id]
                if args.gt_objects:
                    boxesgt_df = vid_df[vid_df.frame == frame_idx]
                    if args.mask_mode != "maskrcnn":
                        boxgtviz.add_boxesgt_viz(
                            ax,
                            boxesgt_df,
                            resize_factor=resize_factor,
                            debug=args.debug,
                        )
                if args.hoa:
                    hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
                    hoaviz.add_hoa_viz(
                        ax, hoa_df, resize_factor=resize_factor, debug=args.debug
                    )
                    if args.mask_mode == "epic":
                        masks_df = masks_dets[masks_dets.frame == frame_idx]
                        masksviz.add_masks_df_viz(
                            ax, masks_df, resize_factor, debug=args.debug
                        )
                    elif args.mask_mode == "maskrcnn":
                        img = np.array(img)
                        with torch.no_grad():
                            res = mask_extractor.masks_from_df(
                                img, hoa_df, resize_factor=resize_factor
                            )
                        labels = [
                            f"{cls}: {score:.2f}"
                            for cls, score in zip(res["classes"], res["scores"])
                        ]
                        masksviz.add_masks_viz(
                            ax,
                            res["masks"],
                            res["boxes"],
                            labels=labels,
                            debug=args.debug,
                        )
                    elif args.mask_mode == "grabcut":
                        img = np.array(img)
                        masks, boxes = grabmasks.masks_from_df(
                            img, hoa_df, resize_factor=resize_factor
                        )
                        masksviz.add_masks_viz(ax, masks, boxes, debug=args.debug)
                    if len(hoa_df) and args.hands:
                        hands_df = handposes.get_hands(
                            hoa_df,
                            img_path=img_path,
                            img_resize_factor=resize_factor,
                            crop_size=256,
                            debug=args.debug,
                        )
                        handviz.add_hand_viz(ax, hands_df)
            else:
                break
            # Get action label time extent bar for given frame
            if args.show_adv:
                adv_colors = labelutils.get_annot_adv(
                    frame_idx, extended_action_labels, cmapping
                )
                ax = fig.add_subplot(2, 1, 1)
                ax.imshow(adv_colors)
                ax.axis("off")
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            if args.debug:
                os.makedirs("tmp", exist_ok=True)
                fig.savefig("tmp/tmp{frame_idx}.png")
            segm_images.append(data)
        # score_clip = mpy.ImageSequenceClip(score_plots, fps=8)
        import pdb; pdb.set_trace()
        clip = mpy.ImageSequenceClip(segm_images, fps=args.fps)
        # final_clip = mpy.clips_array([[clip,], [score_clip,]])
        clip.write_videofile(rendered_path)
        print(f"Saved video to {rendered_path}")
    except Exception:
        traceback.print_exc()
