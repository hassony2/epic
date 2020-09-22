import argparse
import os
import pickle
import tarfile
import yaml

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d, viz2d
from libyana.metrics.iou import get_iou
from libyana.transformutils.handutils import get_affine_transform, transform_img

from epic.boxutils import (
    epic_box_to_norm,
    get_center_scale,
    get_closest,
    extend_props,
    extend_labels,
)
from epic.displayutils import add_load_img
from epic import tarutils, euristics

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit_folder",
        default="/mnt/disks/myssd/code/handpose/obman_train/results/2019_01_22_parallel",
    )
    parser.add_argument(
        "--rec_folder",
        default="/mnt/disks/mydrive/datasets/epic/handrec/2019_01_22/pkls/",
    )
    parser.add_argument(
        "--epic_root",
        default="/mnt/disks/myssd/datasets/epic/frames/epictmp/frames_rgb_flow/rgb/train/",
    )
    parser.add_argument("--fps", default=5, type=int)
    parser.add_argument("--img_freq", default=1, type=int)
    parser.add_argument("--img_ext", default="png")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--use_tar", action="store_true")
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--without_image", action="store_true")
    args = parser.parse_args()
    if args.rec_folder.endswith("/"):
        base_folder = os.path.dirname(args.rec_folder[:-1])

    if args.interpolate:
        suff = "interp"
    else:
        suff = "nointerp"
    if args.rename:
        suff += "rename"
    if args.without_image:
        suff += "withoutimg"
    res_folder = os.path.join(base_folder, f"objects_{args.img_ext}_{suff}")
    print(f"Saving img with bboxes to {res_folder}")
    os.makedirs(res_folder, exist_ok=True)

    fit_folder = os.path.join(args.fit_folder, "results")
    fit_files = [
        os.path.join(fit_folder, rec_file) for rec_file in os.listdir(fit_folder)
    ]
    with open(f"EPIC_{args.split}_object_action_correspondence.pkl", "rb") as p_f:
        corresp_data = pickle.load(p_f)
    # Get object infos
    object_data = pd.read_csv("EPIC_train_object_labels.csv")
    object_data.bounding_boxes = object_data.bounding_boxes.apply(eval)

    # Get annot info
    action_data = training_labels()

    rec_files = [
        os.path.join(args.rec_folder, rec_file)
        for rec_file in os.listdir(args.rec_folder)
    ]
    # Load fits
    last_tar_path = None
    last_video_id = None
    tarf = None
    box_props = None
    fig = plt.figure(figsize=(7, 5))
    resize_factor = 456 / 1920  # from original to target resolution
    for rec_path in sorted(rec_files):
        fit_file = os.path.join(args.rec_folder, os.path.basename(rec_path))
        with open(fit_file, "rb") as p_f:
            fit_results = pickle.load(p_f)
        frames_np = []
        with open(rec_path, "rb") as p_f:
            rec_results = pickle.load(p_f)
        fit_results = fit_results[:: args.img_freq]
        rec_results = rec_results[:: args.img_freq]
        frame_nb = len(rec_results)
        print(frame_nb)
        for rec_result in tqdm(rec_results):
            hands = euristics.get_hand_infos(rec_result)
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            frame_idx = rec_result["frame_idx"]
            frame_name = f"frame_{frame_idx:010d}.jpg"
            video_full_id = rec_result["seq"]
            if box_props is None or video_full_id != last_video_id:
                print("change!")
                # Get correct boxes correspondences
                video_data = object_data[object_data["video_id"] == video_full_id]
                video_corresp_data = corresp_data[
                    (corresp_data["video_id"] == video_full_id)
                ]
                object2action_frame = dict(
                    zip(video_corresp_data["object_frame"], video_corresp_data.index)
                )
                action2object_frame = dict(
                    zip(video_corresp_data.index, video_corresp_data["object_frame"])
                )

                haskeys = [frame in object2action_frame for frame in video_data.frame]
                newkeys = [
                    object2action_frame[frame]
                    for frame in video_data.frame
                    if frame in object2action_frame
                ]
                print(f"keeping {sum(haskeys)} out of {len(haskeys)}")

                video_data = video_data[haskeys]
                video_data.frame = newkeys
                box_props = video_data
                box_props = box_props[box_props.bounding_boxes.astype(bool)]
                box_props = extend_props(box_props, interpolate=args.interpolate)
                video_action_data = action_data[action_data.video_id == video_full_id]
                action_segms = extend_labels(video_action_data)
                last_video_id = video_full_id

            if frame_idx in box_props:
                # Get image
                if args.use_tar:
                    tar_path = os.path.join(
                        args.epic_root, video_full_id[:3], f"{video_full_id}.tar"
                    )
                    # Only reopen if change of tar file
                    if tar_path != last_tar_path:
                        tarf = tarfile.open(tar_path)
                        last_tar_path = tar_path
                    frame_subpath = f"./{frame_name}"
                    frame = tarutils.cv2_imread_tar(tarf, frame_subpath)
                else:
                    video_folder = os.path.join(
                        args.epic_root, video_full_id[:3], video_full_id
                    )
                    frame = cv2.imread(os.path.join(video_folder, frame_name))
                if not args.without_image:
                    ax.imshow(frame[:, :, ::-1])
                else:
                    ax.imshow(np.zeros_like(frame[:, :, ::-1]))
                ax.set_title(action_segms[frame_idx])
                ax.axis("off")

                box_annots = box_props[frame_idx]
                for hand in hands:
                    viz2d.visualize_joints_2d(ax, hand, joint_idxs=False)
                for bboxes, noun in box_annots:
                    bboxes_norm = [
                        epic_box_to_norm(bbox, resize_factor=resize_factor)
                        for bbox in bboxes
                    ]
                    label_color = "w"
                    detect2d.visualize_bboxes(
                        ax,
                        bboxes_norm,
                        labels=[
                            noun,
                        ]
                        * len(bboxes),
                        label_color=label_color,
                    )
                if args.rename:
                    img_name = f"{frame_idx:08d}.{args.img_ext}"
                else:
                    img_name = os.path.basename(
                        rec_path.replace(".pkl", f"{frame_idx:08d}.{args.img_ext}")
                    )
                video_res_folder = os.path.join(res_folder, video_full_id)
                os.makedirs(video_res_folder, exist_ok=True)
                img_path = os.path.join(video_res_folder, img_name)
                print(f"Saving to {img_path}")
                fig.savefig(img_path)
