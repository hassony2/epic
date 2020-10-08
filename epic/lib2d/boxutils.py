import torch


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
