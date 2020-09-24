import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from pyopp.hand import Hand
from epic.hpose import preprocess, postprocess
from libyana.visutils import viz2d
import pandas as pd

hand_opp = Hand("misc/opp/hand_pose_model.pth")


def get_hands(
    hoa_df,
    img_path,
    crop_size=256,
    img_resize_factor=1,
    hand_mode="opp",
    scale_factor=1.5,
    debug=True,
):
    hand_dicts = []
    for det_idx, det in hoa_df.iterrows():
        if det.det_type == "hand":
            det_bbox = [
                val * img_resize_factor
                for val in [det.left, det.top, det.right, det.bottom]
            ]
            square_det = preprocess.squarify(det_bbox, scale_factor)
            img = Image.open(img_path)
            crop = img.crop(square_det)
            crop_radius = square_det[2] - square_det[0]
            # Resize and convert to BGR
            hand_crop = cv2.resize(np.array(crop), (crop_size, crop_size))[:, :, ::-1]
            if det.side == "left":
                # Process all hands as right hands
                joints2d, peak_scores = hand_opp(np.flip(hand_crop, axis=1))
                joints2d = postprocess.flip_coords(
                    joints2d, crop_size=crop_size, axis=0
                )
            else:
                joints2d, peak_scores = hand_opp(hand_crop)
            mean_scores = np.stack(peak_scores).mean(0)[0]

            # joints2d_abs = joints2d / crop_size * crop_radius + np.array([det_bbox[0], det_bbox[1]])
            joints2d_abs = joints2d / crop_size * crop_radius + np.array(
                [square_det[0], square_det[1]]
            )
            if debug:
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                # scatter(joints2d_abs[:, 0], joints2d_abs[:, 1], s=1)
                viz2d.visualize_joints_2d(ax, joints2d_abs, joint_idxs=False)
                fig.savefig("tmp.png")
                plt.clf()
                fig, ax = plt.subplots(1)
                ax.imshow(hand_crop[:, :, ::-1])
                viz2d.visualize_joints_2d(ax, joints2d, joint_idxs=False)
                fig.savefig("tmp_crop.png")
            det_dict = det.to_dict()
            det_dict["joints2d"] = joints2d_abs
            det_dict["joints2d_scores"] = mean_scores
            hand_dicts.append(det_dict)
    return pd.DataFrame(hand_dicts)
