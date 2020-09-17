import argparse
import os
import pickle
import tarfile

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
import moviepy.editor as mpy
from tqdm import tqdm

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d
from libyana.metrics.iou import get_iou
from libyana.transformutils.handutils import get_affine_transform, transform_img

from epic import displayutils
from epic import labelutils
from epic.hoa import gethoa
from epic.viz import hoaviz, boxgtviz
from epic.hpose import handposes, handviz

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train", choices=["train", "test"])
parser.add_argument(
    "--epic_root",
    default="/sequoia/data2/yhasson/datasets/epic-kitchen/process_yana/frames_rgb_flow/rgb_frames/",
)
parser.add_argument("--use_tar", action="store_true")
parser.add_argument("--video_id", default=1, type=int)
parser.add_argument("--person_id", default=1, type=int)
parser.add_argument("--frame_nb", default=100000, type=int)
parser.add_argument("--frame_step", default=10, type=int)
parser.add_argument("--fps", default=8, type=int)
parser.add_argument("--no_objects", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument(
    "--hoa", action="store_true", help="Add predicted hand and object bbox annotations"
)
parser.add_argument(
    "--hoa_root",
    default="/sequoia/data2/dataset/epic-100/3l8eci2oqgst92n14w2yqi5ytu/hand-objects/",
)
args = parser.parse_args()

args.video_id = f"{args.video_id:02d}"
args.person_id = f"P{args.person_id:02d}"

for key, val in vars(args).items():
    print(f"{key}: {val}")

train_labels = training_labels()
video_full_id = f"{args.person_id}_{args.video_id}"
video_action_data = train_labels[(train_labels["video_id"] == video_full_id)]

extended_action_labels, _ = labelutils.extend_action_labels(video_action_data)
action_names = set(extended_action_labels.values())

if not args.no_objects:
    obj_df = labelutils.get_obj_labels(
        video_id=video_full_id, person_id=args.person_id, interpolate=True
    )
if args.hoa:
    hoa_dets = gethoa.load_video_hoa(video_full_id, hoa_root=args.hoa_root)

save_folder = "results/action_videos"
frame_template = os.path.join(args.epic_root, "{}/{}/{}/frame_{:010d}.jpg")
# rendered_path = os.path.join(save_folder, f"{video_full_id}.mp4")
rendered_path = os.path.join(save_folder, f"{video_full_id}.webm")
os.makedirs(save_folder, exist_ok=True)

fig = plt.figure(figsize=(4, 5))
all_images = []
cmapping = displayutils.get_colors(action_names)

resize_factor = 456 / 1920  # from original to target resolution
for frame_idx in tqdm(range(1, args.frame_nb + 1, args.frame_step)):
    frame_name = "frame_{frame_idx:010d}.jpg"
    frame_subpath = f"./{frame_name}"
    fig.clf()
    ax = fig.add_subplot(2, 1, 2)
    img_path = frame_template.format(
        args.split, args.person_id, video_full_id, frame_idx
    )
    if frame_idx in extended_action_labels:
        label = f"fr{frame_idx}_{extended_action_labels[frame_idx]}"
    else:
        label = f"fr{frame_idx}"
    if os.path.exists(img_path):
        displayutils.add_load_img(ax, img_path, label)
        # Display object annotations (ground truth)
        if not args.no_objects:
            boxesgt_df = obj_df[obj_df.frame == frame_idx]
            boxgtviz.add_boxesgt_viz(
                ax, boxesgt_df, resize_factor=resize_factor, debug=args.debug
            )
        if args.hoa:
            hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
            hoaviz.add_hoa_viz(
                ax, hoa_df, resize_factor=resize_factor, debug=args.debug
            )
            if len(hoa_df):
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
    adv_colors = labelutils.get_annot_adv(frame_idx, extended_action_labels, cmapping)
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(adv_colors)
    ax.axis("off")
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    all_images.append(data)
clip = mpy.ImageSequenceClip(all_images, fps=args.fps)
# final_clip = mpy.clips_array([[clip,], [score_clip,]])
clip.write_videofile(rendered_path)
print(f"Saved video to {rendered_path}")
