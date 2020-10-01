import torch


def trans_init_from_boxes(boxes, K, z_range=(0.5, 0.5)):
    # Used in the paper
    assert len(z_range) == 2
    assert boxes.shape[-1] == 4
    assert boxes.dim() == 2
    bsz = boxes.shape[0]
    uv_centers = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2
    z = (
        torch.as_tensor(z_range)
        .mean()
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(bsz, 1)
        .to(boxes.device)
        .to(boxes.dtype)
    )
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    xy_init = ((uv_centers - cxcy) * z) / fxfy
    return torch.cat([xy_init, z], 1)
