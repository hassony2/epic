from pytorch3d import utils as py3dutils

import numpy as np
import torch

from epic.lib3d import rotations as rotutils
from epic.lib3d import normalize

from trimesh import creation as tricreation


class IcoSphere(torch.nn.Module):
    def __init__(
        self,
        batch_size=2,
        radius=0.2,
        camextr=None,
        hand_links=None,
        debug=True,
        random_rot=True,
        z_off=0.5,
        y_off=0,
        x_off=0,
        mesh_type="box",
    ):
        super().__init__()
        self.batch_size = batch_size
        if mesh_type == "box":
            box = tricreation.box([1, 1, 1])
            faces = torch.Tensor(np.array(box.faces))
            verts_loc = torch.Tensor(np.array(box.vertices))
        elif mesh_type == "sphere":
            icomesh = py3dutils.ico_sphere(2)
            verts_loc = icomesh.verts_padded()[0]
            faces = icomesh.faces_padded()[0]
        else:
            raise ValueError(f"{mesh_type} not in [sphere|box]")
        # Normalize and save verts
        norm_verts = normalize.normalize_verts(verts_loc, 1)
        self.register_buffer(
            "verts", norm_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        )
        self.register_buffer(
            "faces", faces.unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # Initialize translation and rotation
        rot_vecs = torch.stack(
            [rotutils.get_rotvec6d(random_rot) for _ in range(batch_size)]
        )

        # Initialize rotation parameters
        self.rot_vecs = torch.nn.Parameter(rot_vecs, requires_grad=True)

        # Initialize scale and translation
        self.trans = torch.nn.Parameter(
            norm_verts.new_zeros(batch_size, 3)
            + norm_verts.new([x_off, y_off, z_off]).view(1, 3),
            requires_grad=True,
        )
        # Scale is shared for all object views
        self.scale = torch.nn.Parameter(
            torch.Tensor([radius]), requires_grad=True
        )

    def get_params(self, optim_scale=False):
        params = [self.trans, self.rot_vecs]
        if optim_scale:
            params.append(self.scale)
        return params

    def forward(self):
        rot_mats = rotutils.compute_rotation_matrix_from_ortho6d(self.rot_vecs)
        verts = self.verts
        faces = self.faces

        # Apply rotation and translation
        rot_verts = verts.bmm(rot_mats)
        rot_verts = rot_verts * self.scale.view(-1, 1, 1).expand(
            self.batch_size, 1, 1
        )
        trans_verts = rot_verts + self.trans.unsqueeze(1)
        return {
            "verts": trans_verts,
            "faces": faces,
            "rot_mats": rot_mats,
            "scales": self.scale,
            "trans": self.trans,
            "rot_vecs": self.rot_vecs,
        }
