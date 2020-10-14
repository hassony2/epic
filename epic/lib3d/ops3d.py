import torch
from pytorch3d import transforms as py3dt


def project(pts, camintr, camextr=None, min_z=None):
    if camextr is not None:
        pts = transform_pts(camextr, pts)
    hom2d = pts.bmm(camintr.permute(0, 2, 1))
    zs = hom2d[:, :, 2:]
    # clamp minimum z to avoid nans
    if min_z is not None:
        zs = torch.max(torch.ones_like(zs) * min_z, zs)
    proj = hom2d[:, :, :2] / zs
    return proj


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


def rot_points(points, centers, axisang=(0, 1, 1)):
    """
    Rotate points around centers
    """
    if points.dim() != 3 or (points.shape[2] != 3):
        raise ValueError(
            "Expected batch of vertices in format (batch_size, vert_nb, 3)"
            f" but got {points.shape}"
        )

    if centers.dim() == 2:
        centers = centers.unsqueeze(1)
    points_c = points - centers
    rot_mats = py3dt.so3_exponential_map(
        points.new(axisang).unsqueeze(0)
    ).view(1, 3, 3)
    points_cr = (
        rot_mats.repeat(points.shape[0], 1, 1)
        .bmm(points_c.transpose(1, 2))
        .transpose(1, 2)
    )
    points_final = points_cr + centers
    return points_final


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


def init_scale_trans_from_boxes_z(
    boxes_2d, K, model_points_3d, zs, camextr=None
):
    """
    Estimates object scale given camera information, bounding box
    and estimated depth
    Assumes object is normalized in sphere of radius 1
    """
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    device = model_points_3d.device
    K = K.to(device)
    boxes_2d = boxes_2d.to(device)
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    # Compute bbox center
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    bb_xy_centers = bb_xy_centers.to(device)
    obj_norms = model_points_3d.norm(2, -1)
    if not torch.allclose(obj_norms.max(), torch.ones_like(obj_norms.max())):
        raise ValueError(
            "Expected model to be normalized to have max norm 1"
            f" but got max norm {obj_norms.max(-1)}"
        )

    if isinstance(zs, torch.Tensor) and (zs.dim() == 1):
        zs = zs.unsqueeze(1).to(device)
    else:
        zs = K.new([[zs]]).repeat(bsz, 1)

    # Prepare model 3D transform
    xy_init = ((bb_xy_centers - cxcy) * zs) / fxfy
    TCO = (
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        .to(torch.float)
        .to(boxes_2d.device)
        .repeat(bsz, 1, 1)
    )
    TCO = TCO.to(device)
    TCO[:, 2, 3:] = zs
    TCO[:, :2, 3] = xy_init.to(device)

    # Compute object 3D and bbox norms
    bb_norm = (boxes_2d[:, 2:] - bb_xy_centers).norm(2, -1)

    # Estimate object scale using scale / bbox_size = z / f
    if isinstance(zs, torch.Tensor):
        scales = zs[:, 0] / fxfy.mean(-1) * bb_norm

    trans = torch.cat([xy_init, zs], 1)
    if camextr is not None:
        # Bring back to world coordinates
        inv_camextr = torch.inverse(camextr)
        trans = (
            inv_camextr[:, :3, :3].bmm(trans.unsqueeze(-1))[:, :3, 0]
            + inv_camextr[:, :3, 3]
        )
    return trans, scales


def trans_init_from_boxes_autodepth(
    boxes_2d, K, model_points_3d, z_guess=0.5, mode="norm", camextr=None
):
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    device = model_points_3d.device
    K = K.to(device)
    boxes_2d = boxes_2d.to(device)
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = (
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        .to(torch.float)
        .to(boxes_2d.device)
        .repeat(bsz, 1, 1)
    )
    TCO[:, 2, 3] = z_guess
    TCO = TCO.to(device)

    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    bb_xy_centers = bb_xy_centers.to(device)

    if isinstance(z_guess, torch.Tensor) and (z_guess.dim() == 1):
        z_guess = z_guess.unsqueeze(1).to(device)
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init.to(device)

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

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    trans = torch.cat([xy_init, z], 1)
    print(trans)
    if camextr is not None:
        # Bring back to world coordinates
        inv_camextr = torch.inverse(camextr)
        trans = (
            inv_camextr[:, :3, :3].bmm(trans.unsqueeze(-1))[:, :3, 0]
            + inv_camextr[:, :3, 3]
        )
    return trans
