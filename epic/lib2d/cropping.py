import torch
import torchvision


def crops(images, boxes, output_size=None):
    assert images.dim() in [3, 4]
    if images.dim() == 3:
        images = images.unsqueeze(1)
    batch_size, _, h, w = images.shape
    device = images.device
    boxes = torch.cat(
        (torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes), dim=1
    )
    crops = torchvision.ops.roi_align(
        images, boxes, output_size=output_size, sampling_ratio=4
    )
    return crops
