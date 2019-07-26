from PIL import Image

from libyana.transformutils.handutils import get_affine_transform, transform_img


def add_load_img(ax, img_path, title=None, transform=None, crop_res=None):
    img = Image.open(img_path)
    if transform is not None:
        img = transform_img(img, transform, crop_res)
    if title is not None:
        ax.set_title(title)
    ax.imshow(img)
    ax.axis('off')
