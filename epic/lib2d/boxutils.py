import torch


def pad(boxes, padding=10):
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
