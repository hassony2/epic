import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as func_transforms


def preprocess_frame(frame):
    # Squarify
    frame = frame[
        :,
        max(0, int(frame.shape[1] / 2 - frame.shape[0] / 2)) : int(
            frame.shape[1] / 2 + frame.shape[0] / 2
        ),
    ]
    frame = cv2.resize(frame, (256, 256))
    return frame


def prepare_input(frame, flip_left_right=False, debug=False):
    # BGR to RGB and flip frame
    input_image = np.flip(frame, axis=2).copy()
    if flip_left_right:
        input_image = np.flip(input_image, axis=1).copy()

    # Concert to shape batch_size=1, rgb, h, w
    input_image = torch.Tensor(input_image.transpose(2, 0, 1))

    # To debug what is actually fed to network
    if debug:
        plt.imshow(input_image.numpy().transpose(1, 2, 0) / 255)
        plt.show()
    input_image = func_transforms.normalize(
        input_image / 255, [0.5, 0.5, 0.5], [1, 1, 1]
    ).unsqueeze(0)
    # Equivalently
    # input_image_1 = input_image / 255 - 0.5
    input_image = input_image.cuda()
    return input_image


def squarify(det, scale_factor=1.5, min_size=None):
    centerx = (det[0] + det[2]) / 2
    centery = (det[1] + det[3]) / 2
    spanx = det[2] - det[0]
    spany = det[3] - det[1]
    span = scale_factor * max(spanx, spany)
    radius = span / 2
    if min_size is not None:
        radius = max(radius, min_size / 2)
    new_det = [centerx - radius, centery - radius, centerx + radius, centery + radius]
    return new_det
