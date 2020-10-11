import torch


def normalize_verts(verts, radius: float = 1):
    """
    Normalize mesh so that it is centered on 0 and is
    inscribed in sphere of radius radius.

    Args:
        radius (float): max distance to average point
    """
    if verts.dim() != 2 or (verts.shape[1] != 3) or (verts.shape[0] == 0):
        raise ValueError(
            f"Expected verts in format (vert_nb, 3), got {verts.shape}"
        )

    # Center object
    verts = verts - verts.mean(0)
    assert torch.allclose(
        verts.mean(0), torch.zeros_like(verts.mean(0)), atol=1e-6
    )

    # Scale
    verts = verts / verts.norm(2, -1).max()
    return verts
