import os

from pytorch3d.io import load_obj as py3dload_obj
import torch

from epic.lib2d import boxutils
from epic.lib3d import rotations as rotutils
from epic.lib3d import normalize, ops3d


class ManipulatedObject(torch.nn.Module):
    def __init__(
        self,
        obj_path,
        bboxes,
        camintr,
        camextr=None,
        hand_links=None,
        debug=True,
        random_rot=True,
        z_off=0.5,
    ):
        super().__init__()
        if not os.path.exists(obj_path):
            # Temp ugly fix, TODO fix
            obj_path = obj_path.replace(
                "/gpfsstore/rech/tan/usk19gv/datasets", "local_data/datasets"
            )
            if not os.path.exists(obj_path):
                raise ValueError(f"Object path {obj_path} does not exist !")
        verts_loc, faces_idx, _ = py3dload_obj(obj_path)

        assert bboxes.ndim == 2 and bboxes.shape[-1] == 4, (
            f"Expected bboxes of shape {bboxes.shape}"
            "to have shape (batch_size, 4)"
        )
        self.boxes = bboxes
        batch_size = len(bboxes)
        self.batch_size = batch_size
        if hand_links is None:
            hand_links = [["right", "left"] for _ in range(batch_size)]
        self.hand_links = hand_links
        # Normalize and save verts
        norm_verts = normalize.normalize_verts(verts_loc, 1)
        self.register_buffer(
            "verts", norm_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        )
        self.register_buffer(
            "faces", faces_idx.verts_idx.unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # Initialize translation and rotation
        rot_vecs = torch.stack(
            [rotutils.get_rotvec6d(random_rot) for box in bboxes]
        )

        batch_camintr = camintr.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_camextr = camextr.unsqueeze(0).repeat(batch_size, 1, 1)

        bboxes = boxutils.preprocess_boxes(
            torch.Tensor(bboxes), padding=10, squarify=True
        )
        verts = norm_verts.unsqueeze(0).repeat(batch_size, 1, 1)
        # Initialize rotation parameters
        self.rot_vecs = torch.nn.Parameter(rot_vecs, requires_grad=True)

        # Initialize scale and translation
        self.trans = torch.nn.Parameter(
            verts.new_zeros(batch_size, 3), requires_grad=True
        )
        # Scale is shared for all object views
        self.scale = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)

        # Initialize objects by depth and bbox size
        self.init_scale_trans_from_depth(z_off, batch_camintr, batch_camextr)

    def init_scale_trans_from_depth(self, z_off, batch_camintr, batch_camextr):
        bboxes = boxutils.preprocess_boxes(
            torch.Tensor(self.boxes), padding=10, squarify=True
        )
        trans, scales = ops3d.init_scale_trans_from_boxes_z(
            bboxes,
            K=batch_camintr,
            model_points_3d=self.verts,
            zs=z_off,
            camextr=batch_camextr,
        )
        self.trans.data[:] = trans
        self.scale.data[:] = scales.mean()

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
