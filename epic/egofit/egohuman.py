import torch
from human_body_prior.tools.model_loader import load_vposer
from epic.smplifyx import (
    vposeutils,
    smplvis,
    smplutils,
)


class EgoHuman(torch.nn.Module):
    def __init__(
        self,
        debug=True,
        batch_size=0,
        hand_pca_nb=6,
        head_center_idx=8949,
        smpl_root="assets/models",
        vposer_dir="assets/vposer",
        vposer_dim=32,
        parts_path="assets/models/smplx/smplx_parts_segm.pkl",
    ):
        super().__init__()
        self.debug = debug
        self.hand_pca_nb = hand_pca_nb
        self.head_center_idx = head_center_idx
        self.smplx_vertex_nb = 10475
        self.vposer_dim = vposer_dim

        # Initialize SMPL-X model
        self.smpl_model = vposeutils.get_smplx(
            model_root=smpl_root, vposer_dir=vposer_dir
        )
        self.smpl_f = self.smpl_model.faces
        # Get vposer
        self.vposer = load_vposer(vposer_dir, vp_model="snapshot")[0]
        self.vposer.eval()

        # Translate human so that head is at camera level
        self.set_head2cam_trans()

        self.armfaces = smplvis.filter_parts(self.smpl_f, parts_path)

        # Initialize model parameters
        self.batch_size = batch_size
        left_hand_pose = self.smpl_model.left_hand_pose.repeat(batch_size, 1)
        right_hand_pose = self.smpl_model.right_hand_pose.repeat(batch_size, 1)
        pose_embedding = (
            self.get_neutral_pose_embedding()
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        self.pose_embedding = torch.nn.Parameter(
            pose_embedding, requires_grad=True
        )
        self.left_hand_pose = torch.nn.Parameter(
            left_hand_pose, requires_grad=True
        )
        self.right_hand_pose = torch.nn.Parameter(
            right_hand_pose, requires_grad=True
        )

    def set_head2cam_trans(self):
        pose_embedding = self.get_neutral_pose_embedding().unsqueeze(0)
        vpose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, pose_embedding
        )
        smplx_verts = self.smpl_model(
            betas=self.smpl_model.betas, body_pose=vpose, return_verts=True
        ).vertices
        head_loc = smplx_verts[:, self.head_center_idx]
        self.smpl_model.transl.data[0] = -(
            head_loc + head_loc.new([0, 0, 0.1])
        )

    def forward(self):
        smpl_params = smplutils.prepare_params(
            self.smpl_model, self.batch_size
        )
        # Override hand and body pose
        smpl_params["right_hand_pose"] = self.right_hand_pose
        smpl_params["left_hand_pose"] = self.left_hand_pose
        body_pose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, self.pose_embedding
        )
        smpl_params["body_pose"] = body_pose
        # Compute body model
        body_model_output = self.smpl_model(
            body_pose=body_pose,
            expression=smpl_params["expression"],
            jaw_pose=smpl_params["jaw_pose"],
            global_orient=smpl_params["global_orient"],
            transl=smpl_params["transl"],
            leye_pose=smpl_params["leye_pose"],
            reye_pose=smpl_params["reye_pose"],
            left_hand_pose=smpl_params["left_hand_pose"],
            right_hand_pose=smpl_params["right_hand_pose"],
            return_verts=True,
            return_full_pose=True,
        )

        res = {
            "faces": self.armfaces.unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
            .to(self.pose_embedding.device),
            "verts": body_model_output.vertices,
            "pose_embedding": self.pose_embedding,
            "body_pose": body_pose,
            "joints": body_model_output.joints,
            "betas": body_model_output.betas,
        }
        return res

    def get_params(
        self,
        param_names=["pose_embedding", "left_hand_pose", "right_hand_pose"],
    ):
        """
        Get parameters to optimize
        """
        params = [
            self.right_hand_pose,
            self.left_hand_pose,
            self.pose_embedding,
        ]
        for param_name, param_vals in self.named_parameters():
            if param_name in param_names:
                param_vals.requires_grad = True
                params.append(param_vals)
            else:
                param_vals.requires_grad = False
        return params

    def get_neutral_pose_embedding(self):
        pose_embedding = torch.Tensor(
            [
                4.4791e-02,
                -2.4548e-01,
                5.0720e-02,
                2.2368e-02,
                1.6534e-02,
                1.0907e-02,
                -2.4805e-01,
                -8.7074e-02,
                2.0321e-01,
                3.6301e-01,
                -2.6616e-01,
                9.5585e-02,
                1.0855e00,
                -9.1609e-03,
                -4.0064e-01,
                8.0400e-02,
                -7.0391e-02,
                -4.7131e-01,
                8.3280e-04,
                -9.0358e-02,
                7.6938e-01,
                -1.6985e-02,
                -2.2199e00,
                -8.9812e-02,
                -8.9908e-02,
                3.3213e-02,
                -4.8088e-01,
                -1.9684e-02,
                -1.7459e-01,
                1.1451e-01,
                -3.0125e-01,
                -1.2514e-01,
            ]
        )
        return pose_embedding
