import numpy as np


def get_sides_eur(frame_infos, img_shape=(456, 256)):
    """Get hand side according to following euristic
    - if only one hand, get side matching half of the image
    - if two hands, assume leftmost is left
    """
    dets = frame_infos["rois"]
    center_x = (dets[:, 1] + dets[:, 3]) / 2
    if dets.shape[0] == 0:
        sides = []
    elif dets.shape[0] == 1:
        # Look if hand is offset in right or left part of the image
        center_off = center_x - img_shape[0] / 2
        if center_off[0] >= 0:
            sides = ["right"]
        else:
            sides = ["left"]
    else:
        if center_x[0] > center_x[1]:
            sides = ["right", "left"]
        else:
            sides = ["left", "right"]
    return sides


def flip_coords(coords, crop_size=256, axis=0):
    coords[:, axis] = crop_size - coords[:, axis]
    return coords


def get_hand_infos(frame_info, img_shape=(456, 256)):
    sides = get_sides_eur(frame_info, img_shape)
    all_hands = []
    for side, hand_info in zip(sides, frame_info["manos"]):
        crop_det = hand_info["square_det"]
        crop_radius = crop_det[2] - crop_det[0]

        # Get right hand
        if side == "right":
            joints2d = hand_info["joints2d"][0]
            joints2d_abs_r = flip_coords(
                joints2d, crop_size=hand_info["crop_size"], axis=0
            ) / hand_info["crop_size"] * crop_radius + np.array([crop_det[0], crop_det[1]])
            all_hands.append(joints2d_abs_r)
        else:
            # Get left hand
            joints2d = hand_info["joints2d"][1]
            joints2d_abs_l = joints2d / hand_info["crop_size"] * crop_radius + np.array(
                [crop_det[0], crop_det[1]]
            )
            all_hands.append(joints2d_abs_l)
    return all_hands
