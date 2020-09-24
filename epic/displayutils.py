from matplotlib.pyplot import cm
import numpy as np
from PIL import Image

from libyana.transformutils.handutils import get_affine_transform, transform_img


def add_load_img(ax, img_path, title=None, transform=None, crop_res=None):
    img = Image.open(img_path)
    if transform is not None:
        img = transform_img(img, transform, crop_res)
    if title is not None:
        ax.set_title(title)
    ax.imshow(img)
    ax.axis("off")
    return img


def get_colors(action_names):
    colors = list(cm.rainbow(np.linspace(0, 1, len(action_names))))
    return dict(zip(action_names, colors))
