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
        debug=True,
        radius=0.1,
        random_rot=True,
        z_off=0.5,
    ):
        super().__init__()
        verts_loc, faces_idx, _ = py3dload_obj(obj_path)

        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (
            f"Expected bboxes of shape {bboxes.shape}"
            "to have shape (batch_size, 4)"
        )
        self.boxes = bboxes
        batch_size = len(bboxes)
        # Normalize and save verts
        norm_verts = normalize.normalize_verts(verts_loc, radius)
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

        bboxes = boxutils.preprocess_boxes(
            torch.Tensor(bboxes), padding=10, squarify=True
        )
        batch_camintr = camintr.unsqueeze(0).repeat(len(bboxes), 1, 1)
        verts = norm_verts.unsqueeze(0).repeat(len(bboxes), 1, 1)
        trans = ops3d.trans_init_from_boxes_autodepth(
            bboxes, batch_camintr, verts, z_guess=z_off
        )
        self.trans = torch.nn.Parameter(trans, requires_grad=True)
        self.rot_vecs = torch.nn.Parameter(rot_vecs, requires_grad=True)
        self.scales = torch.nn.Parameter(
            torch.Tensor([radius for box in bboxes]), requires_grad=True
        )

    def get_params(self, optim_scale=False):
        params = [self.trans, self.rot_vecs]
        if optim_scale:
            params.append(self.scales)
        return params

    def forward(self):
        rot_mats = rotutils.compute_rotation_matrix_from_ortho6d(self.rot_vecs)
        verts = self.verts
        faces = self.faces

        # Apply rotation and translation
        rot_verts = verts.bmm(rot_mats)
        rot_verts = rot_verts * self.scales.view(-1, 1, 1)
        trans_verts = rot_verts + self.trans.unsqueeze(1)
        return {
            "verts": trans_verts,
            "faces": faces,
            "rot_mats": rot_mats,
            "scales": self.scales,
            "trans": self.trans,
            "rot_vecs": self.rot_vecs,
        }
