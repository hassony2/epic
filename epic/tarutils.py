import os

import cv2
import numpy as np


def tar_from_frame_path(frame_path):
    base_path = os.path.dirname(frame_path)
    filename = os.path.basename(frame_path)
    return f"{base_path}.tar", filename


def get_np_array_from_tar_object(tar_extractfl):
    """converts a buffer from a tar file in np.array"""
    return np.asarray(bytearray(tar_extractfl.read()), dtype=np.uint8)


def cv2_imread_tar(tar_ref, filename):
    np_dec = get_np_array_from_tar_object(tar_ref.extractfile(filename))
    frame = cv2.imdecode(np_dec, cv2.IMREAD_COLOR)
    return frame
