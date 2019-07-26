import argparse
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

import numpy as np
import pandas as pd
from PIL import Image
import moviepy.editor as mpy

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d
from libyana.metrics.iou import get_iou
from libyana.transformutils.handutils import get_affine_transform, transform_img

from epic.displayutils import add_load_img

def extend_action_labels(video_action_data):
    dense_annots = {}
    for row_idx, action_row in video_action_data.iterrows():
        start_frame = action_row['start_frame']
        stop_frame = action_row['stop_frame']

        narration = action_row['narration']
        # all_nouns = action_row['all_nouns']
        # noun = action_row['noun']
        # verb = action_row['verb']
        for frame_idx in range(start_frame, stop_frame + 1):
            dense_annots[frame_idx] = narration
    return dense_annots

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='train', choices=['train', 'test'])
parser.add_argument('--video_id', default=1, type=int)
parser.add_argument('--person_id', default=1, type=int)
parser.add_argument('--frame_nb', default=100000, type=int)
args = parser.parse_args()

args.video_id = f'{args.video_id:02d}'
args.person_id = f'P{args.person_id:02d}'

for key, val in vars(args).items():
    print(f'{key}: {val}')

train_labels = training_labels()
video_full_id = f'{args.person_id}_{args.video_id}'
video_action_data = train_labels[(train_labels['video_id'] == video_full_id)]

extended_action_labels = extend_action_labels(video_action_data)
action_names = set(extended_action_labels.values())

save_folder = 'results/action_videos'
frame_template = 'D:/datasets/epic/rgb_frames/{}/{}/{}/frame_{:010d}.jpg'
rendered_path = os.path.join(save_folder, f'{video_full_id}.mp4')
os.makedirs(save_folder, exist_ok=True)

def get_colors(action_names):
    colors = list(cm.rainbow(np.linspace(0, 1, len(action_names))))
    return dict(zip(action_names, colors))


def get_annot_adv(frame_idx, action_labels, cmapping, span=10, extent=7):
    colors = []
    for adv_idx in range(frame_idx - span, frame_idx + span):
        if adv_idx in action_labels:
            action = action_labels[adv_idx]
            color = cmapping[action]
        else:
            color = [0, 0, 0, 1]
        colors.append(color)
    colors = np.array([[color,] * extent for color in colors]).reshape(len(colors) * extent, 4)
    colors = colors[:, :3].transpose().reshape((-1, *colors[:, :3].shape)).repeat(extent, 0)
    return colors


fig = plt.figure()
all_images = []
cmapping = get_colors(action_names)

for frame_idx in range(1, args.frame_nb + 1):
    fig.clf()
    ax = fig.add_subplot(1, 2, 1)
    img_path = frame_template.format(args.split, args.person_id, video_full_id, frame_idx)
    if frame_idx in extended_action_labels:
        label = f'fr{frame_idx}_{extended_action_labels[frame_idx]}'
    else:
        label = f'fr{frame_idx}'
    if os.path.exists(img_path):
        add_load_img(ax, img_path, label)
    else:
        break
    adv_colors = get_annot_adv(frame_idx, extended_action_labels, cmapping)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(adv_colors)
    ax.axis('off')
    # ax.axvline(x=plot_idx)
    # fig.legend(loc='lower left')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    all_images.append(data)
# score_clip = mpy.ImageSequenceClip(score_plots, fps=8)
clip = mpy.ImageSequenceClip(all_images, fps=8)
# final_clip = mpy.clips_array([[clip,], [score_clip,]])
clip.write_videofile(rendered_path)
print(f'Saved video to {rendered_path}')
