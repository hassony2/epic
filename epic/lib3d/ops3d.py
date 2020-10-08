import torch


def transform_pts(T, pts):
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError("Unsupported shape for T", T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


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


def trans_init_from_boxes_autodepth(
    boxes_2d, K, model_points_3d, z_guess=0.5, mode="norm"
):
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = (
        torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z_guess], [0, 0, 0, 1]]
        )
        .to(torch.float)
        .to(boxes_2d.device)
        .repeat(bsz, 1, 1)
    )
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)
    if mode == "dxdy":
        deltax_3d = (
            C_pts_3d[:, :, 0].max(dim=1).values
            - C_pts_3d[:, :, 0].min(dim=1).values
        )
        deltay_3d = (
            C_pts_3d[:, :, 1].max(dim=1).values
            - C_pts_3d[:, :, 1].min(dim=1).values
        )

        bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
        bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

        z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
        z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay
        z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2
    elif mode == "norm":
        delta_norms = C_pts_3d[:, :, :2].norm(2, -1).max(dim=1)[0]
        bb_norm = (boxes_2d[:, 2:] - bb_xy_centers).norm(2, -1)
        z = (fxfy[:, 0] * delta_norms / bb_norm).unsqueeze(1)
        print(z.shape)

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    return torch.cat([xy_init, z], 1)
