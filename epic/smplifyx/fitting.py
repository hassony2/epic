# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings

import numpy as np

import torch
import torch.nn as nn

from epic.smplifyx import utils
from epic.smplifyx import smplutils, vposeutils

# from epic.rendering.py3drendutils import batch_render


class FittingMonitor(object):
    def __init__(
        self,
        summary_steps=1,
        visualize=False,
        maxiters=100,
        ftol=2e-09,
        gtol=1e-05,
        body_color=(1.0, 1.0, 0.9, 1.0),
        model_type="smpl",
        **kwargs,
    ):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3), [batch_size, 1]
        )

    def run_fitting(
        self,
        optimizer,
        closure,
        params,
        body_model,
        use_vposer=True,
        pose_embedding=None,
        vposer=None,
        **kwargs,
    ):
        """Helper function for running an optimization process
        Parameters
        ----------
            optimizer: torch.optim.Optimizer
                The PyTorch optimizer object
            closure: function
                The function used to calculate the gradients
            params: list
                List containing the parameters that will be optimized
            body_model: nn.Module
                The body model PyTorch module
            use_vposer: bool
                Flag on whether to use VPoser (default=True).
            pose_embedding: torch.tensor, BxN
                The tensor that contains the latent pose variable.
            vposer: nn.Module
                The VPoser module
        Returns
        -------
            loss: float
            The final loss value
        """
        prev_loss = None
        losses = []
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print("NaN loss value, stopping!")
                break

            if torch.isinf(loss).sum() > 0:
                print("Infinite loss value, stopping!")
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = np.abs(
                    utils.rel_change(prev_loss, loss.item())
                )

                if loss_rel_change <= self.ftol:
                    break

            if all(
                [
                    torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params
                    if var.grad is not None
                ]
            ):
                break

            prev_loss = loss.item()
            losses.append(prev_loss)
        body_pose = vposeutils.get_vposer_pose(
            body_model, vposer, pose_embedding
        )

        body_model_output = self.smpl_forward(body_model, body_pose)
        verts = body_model_output.vertices.detach().cpu().numpy()
        res = {
            "losses": losses,
            "body_pose": body_pose,
            "embedding": pose_embedding,
            "verts": verts,
        }
        return res

    def smpl_forward(
        self, body_model, body_pose, return_verts=True, return_full_pose=True
    ):
        smpl_params = smplutils.prepare_params(body_model, body_pose.shape[0])
        body_model_output = body_model(
            body_pose=body_pose,
            expression=smpl_params["expression"],
            jaw_pose=smpl_params["jaw_pose"],
            global_orient=smpl_params["global_orient"],
            transl=smpl_params["transl"],
            leye_pose=smpl_params["leye_pose"],
            reye_pose=smpl_params["reye_pose"],
            left_hand_pose=smpl_params["left_hand_pose"],
            right_hand_pose=smpl_params["right_hand_pose"],
            return_verts=return_verts,
            return_full_pose=return_full_pose,
        )
        return body_model_output

    def create_fitting_closure(
        self,
        optimizer,
        body_model,
        show_faces=None,
        model_faces=None,
        camera=None,
        gt_joints=None,
        gt_verts=None,
        imgs=None,
        joints_conf=None,
        joint_weights=None,
        loss=None,
        verts_conf=None,
        vert_weights=None,
        return_verts=True,
        return_full_pose=False,
        use_vposer=False,
        vposer=None,
        pose_embedding=None,
        create_graph=False,
        **kwargs,
    ):
        faces_tensor = body_model.faces_tensor

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()
            body_pose = vposeutils.get_vposer_pose(
                body_model, vposer, pose_embedding
            )
            body_model_output = self.smpl_forward(body_model, body_pose)

            total_loss = loss(
                body_model_output,
                camera=camera,
                gt_joints=gt_joints,
                gt_verts=gt_verts,
                body_model_faces=faces_tensor,
                imgs=imgs,
                joints_conf=joints_conf,
                joint_weights=joint_weights,
                verts_conf=verts_conf,
                vert_weights=vert_weights,
                pose_embedding=pose_embedding,
                show_faces=show_faces,
                use_vposer=use_vposer,
                **kwargs,
            )

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(
                    return_verts=True, body_pose=body_pose
                )
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(), body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type="smplify", **kwargs):
    if loss_type == "smplify":
        return SMPLifyLoss(**kwargs)
    elif loss_type == "camera_init":
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))


class SMPLifyLoss(nn.Module):
    def __init__(
        self,
        search_tree=None,
        pen_distance=None,
        tri_filtering_module=None,
        rho=100,
        body_pose_prior=None,
        shape_prior=None,
        expr_prior=None,
        angle_prior=None,
        jaw_prior=None,
        use_joints_conf=True,
        use_face=False,
        use_hands=True,
        left_hand_prior=None,
        right_hand_prior=None,
        interpenetration=True,
        dtype=torch.float32,
        data_weight=1.0,
        body_pose_weight=0.0,
        shape_weight=0.0,
        bending_prior_weight=0.0,
        hand_prior_weight=0.0,
        expr_prior_weight=0.0,
        jaw_prior_weight=0.0,
        show_faces=None,
        coll_loss_weight=0.0,
        reduction="sum",
        **kwargs,
    ):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer(
            "data_weight", torch.tensor(data_weight, dtype=dtype)
        )
        self.register_buffer(
            "body_pose_weight", torch.tensor(body_pose_weight, dtype=dtype)
        )
        self.register_buffer(
            "shape_weight", torch.tensor(shape_weight, dtype=dtype)
        )
        self.register_buffer(
            "bending_prior_weight",
            torch.tensor(bending_prior_weight, dtype=dtype),
        )
        if self.use_hands:
            self.register_buffer(
                "hand_prior_weight",
                torch.tensor(hand_prior_weight, dtype=dtype),
            )
        if self.use_face:
            self.register_buffer(
                "expr_prior_weight",
                torch.tensor(expr_prior_weight, dtype=dtype),
            )
            self.register_buffer(
                "jaw_prior_weight", torch.tensor(jaw_prior_weight, dtype=dtype)
            )
        if self.interpenetration:
            self.register_buffer(
                "coll_loss_weight", torch.tensor(coll_loss_weight, dtype=dtype)
            )

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if "torch.Tensor" in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(
                        loss_weight_dict[key],
                        dtype=weight_tensor.dtype,
                        device=weight_tensor.device,
                    )
                setattr(self, key, weight_tensor)

    def forward(
        self,
        body_model_output,
        camera,
        gt_joints=None,
        gt_verts=None,
        imgs=None,
        joints_conf=None,
        verts_conf=None,
        body_model_faces=None,
        joint_weights=None,
        vert_weights=None,
        use_vposer=False,
        pose_embedding=None,
        show_faces=None,
        **kwargs,
    ):
        if gt_joints is not None:
            projected_joints = camera(body_model_output.joints)
            # Calculate the weights for each joints
            weights = (
                joint_weights * joints_conf
                if self.use_joints_conf
                else joint_weights
            ).unsqueeze(dim=-1)

            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            # joint_diff = self.robustifier(gt_joints - projected_joints)
            # Dirty hack [:, :67] to skip face
            joint_diff = self.robustifier(gt_joints - projected_joints[:, :67])
            joint_loss = (
                torch.sum(weights ** 2 * joint_diff) * self.data_weight ** 2
            )
        else:
            joint_loss = 0
        if gt_verts is not None:
            projected_verts = camera(body_model_output.vertices)
            weights = (vert_weights * verts_conf).unsqueeze(dim=-1)
            # vert_diff = self.robustifier(gt_verts[:, :, :2] - projected_verts)
            vert_diff = (gt_verts[:, :, :2] - projected_verts) / 100
            # from matplotlib import pyplot as plt
            # fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            # ax = axes[0]
            # pts1 = gt_verts[0].cpu().detach().numpy()
            # super_verts = gt_verts[0][(weights[0]>0).repeat(1, 3)].view(-1, 3)
            # pts2 = (projected_verts[0].cpu()
            #    .detach().numpy()[np.unique(show_faces).astype(np.uint32)])
            # pred_verts = (projected_verts[0][(weights[0]>0).repeat(1, 2)].
            #      view(-1, 2).cpu().detach().numpy())
            # pts3 = body_model_output.vertices.cpu().detach().numpy()[0]
            # ax.imshow(imgs[0])
            # ax.scatter(pts2[:, 0], pts2[:, 1], s=1, alpha=0.2, c="r")
            # from matplotlib import cm
            # point_nb = pred_verts.shape[0]
            # colors = cm.rainbow(np.linspace(0, 1, point_nb))
            # ax.scatter(super_verts[:, 0], super_verts[:, 1],
            #    s=0.5, alpha=0.8, c=colors)
            # ax.scatter(pred_verts[:, 0], pred_verts[:, 1],
            #    s=0.5, alpha=0.2, c=colors)

            # ax = axes[1]
            # ax.scatter(pts3[:, 0], pts3[:, 1], s=1)
            # ax.axis("equal")
            # ax = axes[2]
            # ax.scatter(pts3[:, 0], pts3[:, 2], s=1)
            # ax.axis("equal")
            # ax = axes[3]
            # ax.scatter(pts3[:, 1], pts3[:, 2], s=1, alpha=0.1)
            # ax.axis("equal")
            # # plt.scatter(pts[0, :, 0],pts[0, :, 2], s=1)
            # # plt.scatter(pts[0, :, 1], pts[0, :, 2], s=1)
            # verts = body_model_output.vertices.cuda()
            # batch_size = verts.shape[0]
            # faces_th = (show_faces.unsqueeze(0).
            #    repeat(batch_size, 1, 1).to(verts.device))
            # camintr = (camera.get_camintr().to(verts.device)
            #            .unsqueeze(0).repeat(batch_size, 1, 1))
            # rot = camera.get_camrot().to(verts.device).unsqueeze(0).
            #          repeat(batch_size, 1, 1).to(verts.device)
            # img_size = (imgs[0].shape[1], imgs[0].shape[0])
            # rends = batch_render(verts, faces_th, K=camintr,
            # image_sizes=[img_size], rot=rot, shading="soft")
            # plt.imshow(rends[0].sum(-1).cpu().numpy())
            # # ax.scatter(pts2[:, 0], pts2[:, 1], s=1)
            # save_folder = "tmp_imgs"
            # os.makedirs(save_folder, exist_ok=True)
            # img_nb = len(os.listdir(save_folder))
            # plt.savefig(os.path.join(save_folder, f"tmp_{img_nb:04d}.png"))
            # Calculate the weights for each joints
            # vert_diff = gt_verts[:, :, :2] - projected_verts
            warnings.warn(
                "epic/smplifyx/fitting.py line 436 only 2D supervision"
            )
            vert_loss = torch.sum(
                (weights * vert_diff) ** 2 * self.data_weight
            ) / (torch.sum(weights) + 0.00001)
            # print((vert_diff * weights).abs().max())
        else:
            vert_loss = 0

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (
                pose_embedding.pow(2).sum() * self.body_pose_weight ** 2
            )
        else:
            pprior_loss = (
                torch.sum(
                    self.body_pose_prior(
                        body_model_output.body_pose, body_model_output.betas
                    )
                )
                * self.body_pose_weight ** 2
            )

        shape_loss = (
            torch.sum(self.shape_prior(body_model_output.betas))
            * self.shape_weight ** 2
        )
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = (
            torch.sum(self.angle_prior(body_pose)) * self.bending_prior_weight
        )

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = (
                torch.sum(
                    self.left_hand_prior(body_model_output.left_hand_pose)
                )
                * self.hand_prior_weight ** 2
            )

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = (
                torch.sum(
                    self.right_hand_prior(body_model_output.right_hand_pose)
                )
                * self.hand_prior_weight ** 2
            )

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = (
                torch.sum(self.expr_prior(body_model_output.expression))
                * self.expr_prior_weight ** 2
            )

            if hasattr(self, "jaw_prior"):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(self.jaw_prior_weight)
                    )
                )

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if self.interpenetration and self.coll_loss_weight.item() > 0:
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1, body_model_faces
            ).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight
                    * self.pen_distance(triangles, collision_idxs)
                )
        total_loss = (
            joint_loss
            + vert_loss
            + pprior_loss
            + shape_loss
            + angle_prior_loss
            + pen_loss
            + jaw_prior_loss
            + expression_loss
            + left_hand_prior_loss
            + right_hand_prior_loss
        )
        batch_size = body_model_output.body_pose.shape[0]
        total_loss = total_loss  # TODO put back / batch_size
        print(
            f"total: {total_loss.item():.4f}"
            f"vert: {vert_loss.item():.4f}"
            f"posep: {pprior_loss.item():.4f}"
            f"angle_p: {angle_prior_loss.item():.4f}"
            f"left_p: {left_hand_prior_loss.item():.4f}"
            f"right_p: {right_hand_prior_loss.item():.4f}"
            f"shpe: {shape_loss.item():.4f}"
        )
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):
    def __init__(
        self,
        init_joints_idxs,
        trans_estimation=None,
        reduction="sum",
        data_weight=1.0,
        depth_loss_weight=1e2,
        dtype=torch.float32,
        **kwargs,
    ):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                "trans_estimation",
                utils.to_tensor(trans_estimation, dtype=dtype),
            )
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer(
            "data_weight", torch.tensor(data_weight, dtype=dtype)
        )
        self.register_buffer(
            "init_joints_idxs",
            utils.to_tensor(init_joints_idxs, dtype=torch.long),
        )
        self.register_buffer(
            "depth_loss_weight", torch.tensor(depth_loss_weight, dtype=dtype)
        )

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(
                    loss_weight_dict[key],
                    dtype=weight_tensor.dtype,
                    device=weight_tensor.device,
                )
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs)
            - torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2,
        )
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (
            self.depth_loss_weight.item() > 0
            and self.trans_estimation is not None
        ):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum(
                (camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2)
            )

        return joint_loss + depth_loss
