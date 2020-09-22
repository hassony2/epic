from copy import deepcopy
from pathlib import Path
from epic.masks import io as masksio
import pandas as pd
from functools import lru_cache

import numpy as np
from PIL import Image
from libyana.metrics import iou

from epic.masks import coco


def framedet2dicts(
    det, obj_thresh=0.2, height=1080, width=1920, filter_mode=None, hoa_df=None
):
    res_dict = {"video_id": det.video_id, "frame": det.frame_number}
    dicts = []
    for obj_det in det.objects:
        det_dict = deepcopy(res_dict)
        score = obj_det.score
        if score > obj_thresh:
            det_dict["score"] = score
            det_dict["left"] = obj_det.bbox.left * width
            det_dict["right"] = obj_det.bbox.right * width
            det_dict["top"] = obj_det.bbox.top * height
            det_dict["bottom"] = obj_det.bbox.bottom * height
            det_dict["pred_class"] = obj_det.pred_class
            class_label = coco.class_names[obj_det.pred_class]
            det_dict["label"] = class_label
            # det_dict["mask"] = resize_mask(obj_det.mask, height=height, width=width)
            det_dict["mask"] = obj_det._coco_mask_counts
            if filter_mode is None:
                keep_det = True
            elif filter_mode == "noperson":
                if "person" in class_label:
                    keep_det = False
                else:
                    keep_det = True
            elif filter_mode == "hoaiou":
                frame_hoa_df = hoa_df[hoa_df.frame == det.frame_number]
                frame_obj_df = frame_hoa_df[frame_hoa_df.det_type == "object"]
                if len(frame_obj_df):
                    det_bbox = [
                        det_dict["left"],
                        det_dict["top"],
                        det_dict["right"],
                        det_dict["bottom"],
                    ]
                    keep_det = False
                    for _, obj_row in frame_obj_df.iterrows():
                        oa_bbox = [
                            obj_row.left,
                            obj_row.top,
                            obj_row.right,
                            obj_row.bottom,
                        ]
                        bbox_iou = iou.get_iou(det_bbox, oa_bbox)
                        if bbox_iou > 0.7:
                            keep_det = True
                else:
                    keep_det = False
            else:
                raise ValueError(
                    f"filter_mode {filter_mode} not in [hoaiou|noperson|None]"
                )
            if keep_det:
                dicts.append(det_dict)
    return dicts


# @lru_cache(maxsize=128)
def load_video_masks(video_id, masks_root, filter_mode=None, hoa_df=None):
    """
    Args:
        video_id (str): PXX_XX video id
        masks_root (str): path to mask folder
                \ P01
                    \ P01_01.pkl
                \ ...
    """
    masks_root = Path(masks_root)
    masks_list = masksio.load_detections(masks_root / video_id[:3] / f"{video_id}.pkl")
    all_masks_dicts = []
    for masks_det in masks_list:
        masks_dicts = framedet2dicts(masks_det, filter_mode=filter_mode, hoa_df=hoa_df)
        all_masks_dicts.extend(masks_dicts)
    dat = pd.DataFrame(all_masks_dicts)
    return dat
