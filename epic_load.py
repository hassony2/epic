import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from epic_kitchens.meta import training_labels
from libyana.visutils import detect2d
from libyana.metrics.iou import get_iou
from libyana.transformutils.handutils import get_affine_transform, transform_img


train_labels = training_labels()
with open('EPIC_train_object_action_correspondence.pkl', 'rb') as p_f:
    corresp_data = pickle.load(p_f)
video_id = '01'
participant_id = 'P01'
video_full_id = f'{participant_id}_{video_id}'
video_action_data = train_labels[(train_labels['video_id'] == video_full_id)]

# Get object infos 
object_data = pd.read_csv('EPIC_train_object_labels.csv')
object_data.bounding_boxes = object_data.bounding_boxes.apply(eval)
object_data = object_data[object_data['video_id'] == video_full_id]

# Get correct boxes correspondences
video_corresp_data = corresp_data[(corresp_data['video_id'] == video_full_id)]
object2action_frame = dict(zip(video_corresp_data['object_frame'], video_corresp_data.index))
object_data.frame = [object2action_frame[frame] for frame in object_data.frame]

crop_res = [256, 256]

def epic_box_to_norm(bbox, resize_factor=1):
    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    bbox = [bbox[0] * resize_factor, bbox[1] * resize_factor, (bbox[0] + bbox[2]) * resize_factor, (bbox[1] + bbox[3])* resize_factor]
    return bbox

def get_center_scale(bbox, scale_factor=1):
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * scale_factor
    return center, scale

def get_closest(value, candidates):
    val_dists = np.abs(candidates - value)
    dist = val_dists.min()
    return (dist, candidates[val_dists.argmin()])

def interpolate_bboxes(start_frame, end_frame, start_bbox, end_bbox):
    inter_boxes = {}
    all_inter_vals = []
    for start_val, end_val in zip(start_bbox, end_bbox):
        inter_vals = np.linspace(start_val, end_val, end_frame - start_frame)
        all_inter_vals.append(inter_vals)
    all_inter_vals = np.array(all_inter_vals).transpose()
    for step_idx in range(0, end_frame - start_frame - 1):
        frame_idx = step_idx + start_frame + 1
        inter_boxes[frame_idx] = (all_inter_vals[step_idx])
    return inter_boxes

def extend_props(candidates):
    annot_frames = candidates['frame'].values
    annot_nouns = candidates['noun'].values
    extended_bboxes = {}
    for annot_idx, (annot_frame, annot_noun) in enumerate(zip(annot_frames[:-1], annot_nouns[:-1])):
        next_frame = annot_frames[annot_idx + 1]
        next_noun = annot_nouns[annot_idx + 1]
        annot_boxes = candidates[candidates.frame == annot_frame].bounding_boxes.values[0]
        # if len(annot_boxes) > 1:
        #     raise ValueError('TODO handle several boxes')
        next_boxes = candidates[candidates.frame == next_frame].bounding_boxes.values[0]

        # Detect consecutive annots with same nouns
        extended_bboxes[annot_frame] = (annot_boxes, annot_noun)
        extended_bboxes[next_frame] = (next_boxes, next_noun)
        if next_frame - annot_frame < 31 and next_noun == annot_noun:
            ious = np.zeros((len(annot_boxes), len(next_boxes)))
            for current_idx, annot_box in enumerate(annot_boxes):
                for next_idx, next_box in enumerate(next_boxes):
                    ious[current_idx, next_idx] = get_iou(epic_box_to_norm(annot_box), epic_box_to_norm(next_box))
            best_next_match_idxs = ious.argmax(1)
            all_inter_boxes = []
            for annot_idx, annot_box in enumerate(annot_boxes):
                inter_bboxes = interpolate_bboxes(annot_frame, next_frame, annot_box, next_boxes[best_next_match_idxs[annot_idx]])
                all_inter_boxes.append(inter_bboxes)
            dict_inter_boxes = {}
            for key in all_inter_boxes[0].keys():
                dict_inter_boxes[key] = ([inter_boxes[key] for inter_boxes in all_inter_boxes], annot_noun)

            extended_bboxes.update(dict_inter_boxes)
    return extended_bboxes

def add_load_img(ax, img_path, title=None, transform=None):
    img = Image.open(img_path)
    if transform is not None:
        img = transform_img(img, affine_trans, crop_res)
    if title is not None:
        ax.set_title(title)
    ax.imshow(img)
    ax.axis('off')

frame_template = '/mnt/d/datasets/epic/rgb_frames/train/{}/{}/frame_{:010d}.jpg'
frame_template = 'D:/datasets/epic/rgb_frames/train/{}/{}/frame_{:010d}.jpg'

for row_idx, action_row in video_action_data[5:].iterrows():
    start_frame = action_row['start_frame']

    narration = action_row['narration']
    all_nouns = action_row['all_nouns']
    stop_frame = action_row['stop_frame']
    noun = action_row['noun']
    verb = action_row['verb']
    box_props = object_data[object_data['noun'] == noun]
    box_props = box_props[box_props.bounding_boxes.astype(bool)]
    all_props = extend_props(box_props)
        start_bbox_img = Image.open(start_bbox_img_path)
    if not len(all_props):
        print(f'Skipping action {row_idx} {verb} {noun}')
    else:
        start_dist, start_clos_value = get_closest(start_frame, np.array(list(all_props.keys())))
        stop_dist, stop_clos_value = get_closest(stop_frame, np.array(list(all_props.keys())))
        print(f'Got match bboxes! Distances: start: {start_dist}, end: {stop_dist}')
        bboxes_start, noun_start = all_props[start_clos_value]
        bboxes_stop, noun_stop = all_props[stop_clos_value]
        fig, axes = plt.subplots(2,3)

        # Load first image
        start_img_path = frame_template.format(participant_id, video_full_id, start_frame)
        add_load_img(axes[0, 0], start_img_path, f'start img')

        # Load closest neighbor with bbox
        start_bbox_img_path = frame_template.format(participant_id, video_full_id, start_clos_value)
        add_load_img(axes[0, 1], start_bbox_img_path, f'closest with bbox: {start_dist}')
        resize_factor = 456/1920  # from original to target resolution
        bboxes_start_norm = [epic_box_to_norm(bbox_start, resize_factor=resize_factor) for bbox_start in bboxes_start]
        print(bboxes_start_norm)
        detect2d.visualize_bboxes(axes[0, 1], bboxes_start_norm, labels=[noun,] * len(bboxes_stop))

        # Crop start_bbox to bbox
        start_bbox = bboxes_start_norm[0]
        center, scale = get_center_scale(start_bbox, scale_factor=2)
        affine_trans, _ = get_affine_transform(center, scale, crop_res)
        add_load_img(axes[0, 2], start_bbox_img_path, title=f'crop', transform=affine_trans)

        # Load last image
        stop_img_path = frame_template.format(participant_id, video_full_id, stop_frame)
        add_load_img(axes[1, 0], stop_img_path, f'stop img')

        # Load closest neighbor with bbox
        stop_bbox_img_path = frame_template.format(participant_id, video_full_id, stop_clos_value)
        add_load_img(axes[1, 1], stop_bbox_img_path, f'closest with bbox: {stop_dist}')
        bboxes_stop_norm = [epic_box_to_norm(bbox_stop, resize_factor=resize_factor) for bbox_stop in bboxes_stop]
        detect2d.visualize_bboxes(axes[1, 1], bboxes_stop_norm, labels=[noun,] * len(bboxes_stop))
        fig.suptitle(f'{verb} {noun}')

        # Crop stop_bbox to bbox
        stop_bbox = bboxes_stop_norm[0]
        center, scale = get_center_scale(stop_bbox, scale_factor=2)
        affine_trans, _ = get_affine_transform(center, scale, crop_res)
        add_load_img(axes[1, 2], stop_bbox_img_path, title=f'crop', transform=affine_trans)

        plt.show()

