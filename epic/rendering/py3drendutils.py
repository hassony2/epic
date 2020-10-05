import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    BlendParams,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import textures


def batch_render(
    verts,
    faces,
    faces_per_pixel=10,
    K=None,
    colors=None,
    color = (0.53, 0.53, 0.8),  # light_purple
    # color = (0.74117647, 0.85882353, 0.65098039),  # light_blue
    image_sizes=None,
    out_res=512,
    bin_size=0,
    shading="soft",
    mode="rgb",
):
    device = torch.device("cuda:0")
    width, height = image_sizes[0]
    out_size = int(max(image_sizes[0]))
    raster_settings = RasterizationSettings(
        image_size=out_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
    )

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    focals = torch.stack([fx, fy], 1)
    px = K[:, 0, 2]
    py = K[:, 1, 2]
    principal_point = torch.stack([width - px, height - py], 1)
    cameras = PerspectiveCameras(
        device=device,
        focal_length=focals,
        principal_point=principal_point,
        image_size=[(out_size, out_size) for _ in range(len(verts))],
    )
    if mode == "rgb" and shading == "soft":
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        lights = DirectionalLights(device=device)
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    elif mode == "silh":
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        shader = SoftSilhouetteShader(blend_params=blend_params)
    elif shading == "faceidx":
        shader = FaceIdxShader()
    elif shading == "facecolor":
        shader = FaceColorShader(face_colors=faces)
    else:
        raise ValueError(f"{shading} not in [facecolor|faceidx|soft]")

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader,
    )
    if mode == "rgb":
        if colors is None:
            colors = torch.from_numpy(np.array(color)).view(1, 1, 3).float().cuda().repeat(1, verts.shape[1], 1)
        tex = textures.TexturesVertex(verts_features=colors)

        meshes = Meshes(verts=verts, faces=faces, textures=tex)
    elif mode == "silh":
        meshes = Meshes(verts=verts, faces=faces)
    else:
        raise ValueError(f"Render mode {mode} not in [rgb|silh]")

    square_images = renderer(meshes, cameras=cameras)
    height_off = int(width - height)
    # from matplotlib import pyplot as plt
    # plt.imshow(square_images.cpu()[0, :, :, 0])
    # plt.savefig("tmp.png")
    images = torch.flip(
        square_images,
        (
            1,
            2,
        ),
    )[:, height_off:]
    return images


class FaceColorShader(torch.nn.Module):
    def __init__(self, face_colors=None, device="cpu"):
        super().__init__()
        batch_s, face_nb, color_nb = face_colors.shape
        vert_face_colors = face_colors.unsqueeze(2).repeat(1, 1, 3, 1)
        self.face_colors = vert_face_colors.view(batch_s * face_nb, 3, color_nb).float()

    def forward(self, fragments, meshes, **kwargs):

        colors = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, self.face_colors
        )
        return colors[:, :, :, 0]


class FaceIdxShader(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        return fragments.pix_to_face
