import cv2
import numpy as np
import torch


def flip_coords(coords, crop_size=256, axis=0):
    coords[:, axis] = crop_size - coords[:, axis]
    return coords


def eul2rotmat(rot_eul):
    if np.linalg.norm(rot_eul):
        R = cv2.Rodrigues(rot_eul)[0]
        rot = torch.Tensor(R).unsqueeze(0)
    else:
        rot = torch.eye(3).unsqueeze(0)
    return rot
