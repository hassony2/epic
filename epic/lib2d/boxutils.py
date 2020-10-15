import torch


def squarify_box(box, scale_factor=1, min_size=None):
    """
    Args:
        box (list): [min_x, min_y, max_x, max_y]
        scale_factor (float): controls margin around original tight box
    Returns:
        (list): Square box [min_x, min_y, max_x, max_y]
    """
    centerx = (box[0] + box[2]) / 2
    centery = (box[1] + box[3]) / 2
    spanx = box[2] - box[0]
    spany = box[3] - box[1]
    span = scale_factor * max(spanx, spany)
    radius = span / 2
    if min_size is not None:
        radius = max(radius, min_size / 2)
    new_box = [
        centerx - radius,
        centery - radius,
        centerx + radius,
        centery + radius,
    ]
    return new_box


def preprocess_boxes(boxes, squarify=True, padding=10):
    if squarify:
        boxes = squarify_boxes(boxes)
    if padding:
        boxes = pad_boxes(boxes, padding)
    return boxes


def squarify_boxes(boxes):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    scales = ((boxes[:, 2:] - boxes[:, :2]) / 2).abs().max(1)[0]
    square_boxes = torch.cat(
        [centers - scales.unsqueeze(1), centers + scales.unsqueeze(1)], 1
    )
    return square_boxes


def pad_boxes(boxes, padding=10):
    new_boxes = torch.stack(
        [
            boxes[:, 0] - padding,
            boxes[:, 1] - padding,
            boxes[:, 2] + padding,
            boxes[:, 3] + padding,
        ],
        1,
    )
    return new_boxes
