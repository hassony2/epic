import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d

object_data = pd.read_csv('EPIC_train_object_labels.csv')
object_data.bounding_boxes = object_data.bounding_boxes.apply(eval)

train_labels = training_labels()
with open('EPIC_train_object_action_correspondence.pkl', 'rb') as p_f:
    corresp_data = pickle.load(p_f)
video_id = '01'
participant_id = 'P01'
video_full_id = f'{participant_id}_{video_id}'
video_corresp_data = corresp_data[(corresp_data['video_id'] == video_full_id)]
video_action_data = train_labels[(train_labels['video_id'] == video_full_id)]

def get_closest(value, candidates):
    val_dists = np.abs(candidates - value)
    dist = val_dists.min()
    return (dist, candidates[val_dists.argmin()])

def interpolate_bboxes(start_frame, end_frame, start_bbox, end_bbox, noun):
    inter_boxes = {}
    all_inter_vals = []
    for start_val, end_val in zip(start_bbox, end_bbox):
        inter_vals = np.linspace(start_val, end_val, end_frame - start_frame)
        all_inter_vals.append(inter_vals)
    all_inter_vals = np.array(all_inter_vals).transpose()
    for step_idx in range(0, end_frame - start_frame - 1):
        frame_idx = step_idx + start_frame + 1
        inter_boxes[frame_idx] = (all_inter_vals[step_idx], noun)
    return inter_boxes

def extend_props(candidates):
    annot_frames = candidates['frame'].values
    annot_nouns = candidates['noun'].values
    extended_bboxes = {}
    for annot_idx, (annot_frame, annot_noun) in enumerate(zip(annot_frames[:-1], annot_nouns[:-1])):
        next_frame = annot_frames[annot_idx + 1]
        next_noun = annot_nouns[annot_idx + 1]
        annot_boxes = candidates[candidates.frame == annot_frame].bounding_boxes.values[0]
        if len(annot_boxes) > 1:
            raise ValueError('TODO handle several boxes')
        annot_box = annot_boxes[0]
        next_box = candidates[candidates.frame == next_frame].bounding_boxes.values[0][0]

        # Detect consecutive annots with same nouns
        extended_bboxes[annot_frame] = (annot_box, annot_noun)
        extended_bboxes[next_frame] = (next_box, next_noun)
        if next_frame - annot_frame < 31 and next_noun == annot_noun:
            inter_bboxes = interpolate_bboxes(annot_frame, next_frame, annot_box, next_box, noun=annot_noun)
            extended_bboxes.update(inter_bboxes)
    return extended_bboxes

def add_load_img(ax, img_path, title=None):
    img = Image.open(img_path)
    ax.imshow(img)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')

frame_template = '/mnt/d/datasets/epic/rgb_frames/train/{}/{}/frame_{:010d}.jpg'
frame_template = 'D:/datasets/epic/rgb_frames/train/{}/{}/frame_{:010d}.jpg'

print(video_action_data)
for row_idx, action_row in video_action_data[5:].iterrows():
    start_frame = action_row['start_frame']
    narration = action_row['narration']
    all_nouns = action_row['all_nouns']
    stop_frame = action_row['stop_frame']
    noun = action_row['noun']
    verb = action_row['verb']
    box_props = object_data[object_data['video_id'] == video_full_id][object_data['noun'] == noun]
    box_props = box_props[box_props.bounding_boxes.astype(bool)]
    all_props = extend_props(box_props)
    if not len(all_props):
        print(f'Skipping action {row_idx} {verb} {noun}')
    else:
        start_dist, start_clos_value = get_closest(start_frame, np.array(list(all_props.keys())))
        stop_dist, stop_clos_value = get_closest(stop_frame, np.array(list(all_props.keys())))
        print(f'Got match bboxes! Distances: start: {start_dist}, end: {stop_dist}')
        bbox_start, noun_start = all_props[start_clos_value]
        bbox_stop, noun_stop = all_props[stop_clos_value]
        fig, axes = plt.subplots(2,2)

        # Load first image
        start_img_path = frame_template.format(participant_id, video_full_id, start_frame)
        add_load_img(axes[0, 0], start_img_path, f'start img: fr {start_frame}')

        # Load closest neighbor with bbox
        start_bbox_img_path = frame_template.format(participant_id, video_full_id, start_clos_value)
        add_load_img(axes[0, 1], start_bbox_img_path, f'closest img with bbox: fr {start_clos_value}')
        resize_factor = 456/1920  # from original to target resolution
        bbox_start = [bbox_start[1], bbox_start[0], bbox_start[3], bbox_start[2]]
        print(bbox_start)
        bbox_start_norm = [bbox_start[0] * resize_factor, bbox_start[1] * resize_factor, (bbox_start[0] + bbox_start[2])* resize_factor, (bbox_start[1] + bbox_start[3])* resize_factor]
        print(bbox_start_norm)
        detect2d.visualize_bbox(axes[0, 1], bbox_start_norm, label=noun)

        # Load last image
        stop_img_path = frame_template.format(participant_id, video_full_id, stop_frame)
        add_load_img(axes[1, 0], stop_img_path, f'stop img: fr {stop_frame}')

        # Load closest neighbor with bbox
        stop_bbox_img_path = frame_template.format(participant_id, video_full_id, stop_clos_value)
        add_load_img(axes[1, 1], stop_bbox_img_path, f'closest img with bbox: fr {stop_clos_value}')
        bbox_stop = [bbox_stop[1], bbox_stop[0], bbox_stop[3], bbox_stop[2]]
        bbox_stop_norm = [bbox_stop[0] * resize_factor, bbox_stop[1] * resize_factor, (bbox_stop[0] + bbox_stop[2])* resize_factor, (bbox_stop[1] + bbox_stop[3])* resize_factor]
        detect2d.visualize_bbox(axes[1, 1], bbox_stop_norm, label=noun)
        fig.suptitle(f'{verb} {noun}')

        plt.show()

