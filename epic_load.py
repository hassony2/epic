import argparse
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d
from libyana.transformutils.handutils import get_affine_transform

from epic.boxutils import (
    epic_box_to_norm,
    get_center_scale,
    get_closest,
    extend_props,
)
from epic.displayutils import add_load_img


parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument("--video_id", default=1, type=int)
parser.add_argument("--person_id", default=1, type=int)
args = parser.parse_args()

args.video_id = f"{args.video_id:02d}"
args.person_id = f"P{args.person_id:02d}"

for key, val in vars(args).items():
    print(f"{key}: {val}")

train_labels = training_labels()
with open(f"EPIC_{args.split}_object_action_correspondence.pkl", "rb") as p_f:
    corresp_data = pickle.load(p_f)
video_full_id = f"{args.person_id}_{args.video_id}"
video_action_data = train_labels[(train_labels["video_id"] == video_full_id)]

# Get object infos
object_data = pd.read_csv("EPIC_train_object_labels.csv")
object_data.bounding_boxes = object_data.bounding_boxes.apply(eval)
object_data = object_data[object_data["video_id"] == video_full_id]

# Get correct boxes correspondences
video_corresp_data = corresp_data[(corresp_data["video_id"] == video_full_id)]
object2action_frame = dict(
    zip(video_corresp_data["object_frame"], video_corresp_data.index)
)
object_data.frame = [object2action_frame[frame] for frame in object_data.frame]

crop_res = [256, 256]
save_folder = "results/images"

frame_template = (
    "/mnt/disks/myssd/datasets/epic/frames/"
    "epictmp/frames_rgb_flow/rgb/{}/{}/{}/frame_{:010d}.jpg"
)

for row_idx, action_row in video_action_data.iterrows():
    start_frame = action_row["start_frame"]

    narration = action_row["narration"]
    all_nouns = action_row["all_nouns"]
    stop_frame = action_row["stop_frame"]
    noun = action_row["noun"]
    verb = action_row["verb"]
    box_props = object_data[object_data["noun"] == noun]
    box_props = box_props[box_props.bounding_boxes.astype(bool)]
    all_props = extend_props(box_props)
    if not len(all_props):
        print(f"Skipping action {row_idx} {verb} {noun}")
    else:
        start_dist, start_clos_value = get_closest(
            start_frame, np.array(list(all_props.keys()))
        )
        stop_dist, stop_clos_value = get_closest(
            stop_frame, np.array(list(all_props.keys()))
        )
        print(
            f"Got match bboxes! Distances: start: {start_dist}, end: {stop_dist}"
        )
        bboxes_start, noun_start = all_props[start_clos_value]
        bboxes_stop, noun_stop = all_props[stop_clos_value]
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))

        # Load first image
        start_img_path = frame_template.format(
            args.split, args.person_id, video_full_id, start_frame
        )
        add_load_img(axes[0, 0], start_img_path, "start img")

        # Load closest neighbor with bbox
        start_bbox_img_path = frame_template.format(
            args.split, args.person_id, video_full_id, start_clos_value
        )
        add_load_img(
            axes[0, 1], start_bbox_img_path, f"closest with bbox: {start_dist}"
        )
        resize_factor = 456 / 1920  # from original to target resolution
        bboxes_start_norm = [
            epic_box_to_norm(bbox_start, resize_factor=resize_factor)
            for bbox_start in bboxes_start
        ]
        detect2d.visualize_bboxes(
            axes[0, 1], bboxes_start_norm, labels=[noun] * len(bboxes_start)
        )

        # Crop start_bbox to bbox
        start_bbox = bboxes_start_norm[0]
        center, scale = get_center_scale(start_bbox, scale_factor=2)
        affine_trans, _ = get_affine_transform(center, scale, crop_res)
        add_load_img(
            axes[0, 2],
            start_bbox_img_path,
            title="crop",
            transform=affine_trans,
            crop_res=crop_res,
        )

        # Load last image
        stop_img_path = frame_template.format(
            args.split, args.person_id, video_full_id, stop_frame
        )
        add_load_img(axes[1, 0], stop_img_path, "stop img")

        # Load closest neighbor with bbox
        stop_bbox_img_path = frame_template.format(
            args.split, args.person_id, video_full_id, stop_clos_value
        )
        add_load_img(
            axes[1, 1], stop_bbox_img_path, f"closest with bbox: {stop_dist}"
        )
        bboxes_stop_norm = [
            epic_box_to_norm(bbox_stop, resize_factor=resize_factor)
            for bbox_stop in bboxes_stop
        ]
        detect2d.visualize_bboxes(
            axes[1, 1], bboxes_stop_norm, labels=[noun] * len(bboxes_stop)
        )
        fig.suptitle(f"{verb} {noun}")

        # Crop stop_bbox to bbox
        stop_bbox = bboxes_stop_norm[0]
        center, scale = get_center_scale(stop_bbox, scale_factor=2)
        affine_trans, _ = get_affine_transform(center, scale, crop_res)
        add_load_img(
            axes[1, 2],
            stop_bbox_img_path,
            title="crop",
            transform=affine_trans,
            crop_res=crop_res,
        )
        save_verb_folder = os.path.join(save_folder, verb)
        os.makedirs(save_verb_folder, exist_ok=True)
        fig.savefig(
            os.path.join(
                save_verb_folder, f"{video_full_id}_{row_idx:06d}_{noun}.png"
            )
        )
