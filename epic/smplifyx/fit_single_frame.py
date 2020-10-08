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


from collections import defaultdict
import pickle
import os
import os.path as osp
import time

import torch

from tqdm import tqdm


from mano_train.smplifyx import optim_factory

from mano_train.smplifyx import fitting
from human_body_prior.tools.model_loader import load_vposer
from mano_train.egofit import vposeutils


def fit_frames(
    img,
    keypoints,
    body_model,
    camera=None,
    joint_weights=None,
    body_pose_prior=None,
    jaw_prior=None,
    left_hand_prior=None,
    right_hand_prior=None,
    shape_prior=None,
    expr_prior=None,
    angle_prior=None,
    loss_type="smplify",
    use_cuda=True,
    init_joints_idxs=(9, 12, 2, 5),
    use_face=False,
    use_hands=True,
    data_weights=None,
    body_pose_prior_weights=None,
    hand_pose_prior_weights=None,
    jaw_pose_prior_weights=None,
    shape_weights=None,
    expr_weights=None,
    hand_joints_weights=None,
    face_joints_weights=None,
    depth_loss_weight=1e2,
    interpenetration=False,
    coll_loss_weights=None,
    df_cone_height=0.5,
    penalize_outside=True,
    max_collisions=8,
    point2plane=False,
    part_segm_fn="",
    focal_length=5000.0,
    side_view_thsh=25.0,
    rho=100,
    vposer_latent_dim=32,
    vposer_ckpt="",
    use_joints_conf=False,
    interactive=True,
    visualize=False,
    batch_size=1,
    dtype=torch.float32,
    ign_part_pairs=None,
    left_shoulder_idx=2,
    right_shoulder_idx=5,
    freeze_camera=True,
    **kwargs
):
    # assert batch_size == 1, "PyTorch L-BFGS only supports batch_size == 1"

    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]
    if data_weights is None:
        data_weights = [1] * len(body_pose_prior_weights)

    msg = "Number of Body pose prior weights {}".format(
        len(body_pose_prior_weights)
    ) + " does not match the number of data term weights {}".format(
        len(data_weights)
    )
    assert len(data_weights) == len(body_pose_prior_weights), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, 0.5 * 1e1]
        msg = (
            "Number of Body pose prior weights does not match the"
            + " number of hand pose prior weights"
        )
        assert len(hand_pose_prior_weights) == len(
            body_pose_prior_weights
        ), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = (
                "Number of Body pose prior weights does not match the"
                + " number of hand joint distance weights"
            )
            assert len(hand_joints_weights) == len(
                body_pose_prior_weights
            ), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, 0.5 * 1e1]
    msg = (
        "Number of Body pose prior weights = {} does not match the"
        + " number of Shape prior weights = {}"
    )
    assert len(shape_weights) == len(body_pose_prior_weights), msg.format(
        len(shape_weights), len(body_pose_prior_weights)
    )

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = (
        "Number of Body pose prior weights does not match the"
        + " number of collision loss weights"
    )
    assert len(coll_loss_weights) == len(body_pose_prior_weights), msg

    use_vposer = kwargs.get("use_vposer", True)
    vposer, pose_embedding = [None] * 2
    if use_vposer:
        pose_embedding = torch.zeros(
            [batch_size, vposer_latent_dim],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model="snapshot")
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = (
            vposeutils.get_vposer_mean_pose().detach().cpu().numpy()
        )
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(
            keypoint_data.shape[0], -1
        )
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, "Interpenetration term can only be used with CUDA"
        assert torch.cuda.is_available(), (
            "No CUDA Device! Interpenetration term can only be used"
            + " with CUDA"
        )

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
            sigma=df_cone_height,
            point2plane=point2plane,
            vectorized=True,
            penalize_outside=penalize_outside,
        )

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, "rb") as faces_parents_file:
                face_segm_data = pickle.load(
                    faces_parents_file, encoding="latin1"
                )
            faces_segm = face_segm_data["segm"]
            faces_parents = face_segm_data["parents"]
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm,
                faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs,
            ).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {
        "data_weight": data_weights,
        "body_pose_weight": body_pose_prior_weights,
        "shape_weight": shape_weights,
    }
    if use_face:
        opt_weights_dict["face_weight"] = face_joints_weights
        opt_weights_dict["expr_prior_weight"] = expr_weights
        opt_weights_dict["jaw_prior_weight"] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict["hand_weight"] = hand_joints_weights
        opt_weights_dict["hand_prior_weight"] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict["coll_loss_weight"] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [
        dict(zip(keys, vals))
        for vals in zip(
            *(
                opt_weights_dict[k]
                for k in keys
                if opt_weights_dict[k] is not None
            )
        )
    ]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(
                weight_list[key], device=device, dtype=dtype
            )

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    # Hand joints start at 25 (before body)
    loss = fitting.create_loss(
        loss_type=loss_type,
        joint_weights=joint_weights,
        rho=rho,
        use_joints_conf=use_joints_conf,
        use_face=use_face,
        use_hands=use_hands,
        vposer=vposer,
        pose_embedding=pose_embedding,
        body_pose_prior=body_pose_prior,
        shape_prior=shape_prior,
        angle_prior=angle_prior,
        expr_prior=expr_prior,
        left_hand_prior=left_hand_prior,
        right_hand_prior=right_hand_prior,
        jaw_prior=jaw_prior,
        interpenetration=interpenetration,
        pen_distance=pen_distance,
        search_tree=search_tree,
        tri_filtering_module=filter_faces,
        dtype=dtype,
        **kwargs
    )
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
        batch_size=batch_size, visualize=visualize, **kwargs
    ) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H
        orientations = [body_model.global_orient.detach().cpu().numpy()]

        # # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(
            tqdm(orientations, desc="Orientation")
        ):
            opt_start = time.time()

            new_params = defaultdict(
                global_orient=orient, body_pose=body_mean_pose
            )
            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(
                tqdm(opt_weights, desc="Stage")
            ):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params)
                )

                if use_vposer:
                    final_params.append(pose_embedding)

                (
                    body_optimizer,
                    body_create_graph,
                ) = optim_factory.create_optimizer(final_params, **kwargs)
                body_optimizer.zero_grad()

                curr_weights["data_weight"] = data_weight
                curr_weights["bending_prior_weight"] = (
                    3.17 * curr_weights["body_pose_weight"]
                )
                if use_hands:
                    # joint_weights[:, 25:67] = curr_weights['hand_weight']
                    pass
                if use_face:
                    joint_weights[:, 67:] = curr_weights["face_weight"]
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer,
                    body_model,
                    camera=camera,
                    gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss,
                    create_graph=body_create_graph,
                    use_vposer=use_vposer,
                    vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True,
                    return_full_pose=True,
                )

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure,
                    final_params,
                    body_model,
                    pose_embedding=pose_embedding,
                    vposer=vposer,
                    use_vposer=use_vposer,
                )

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write(
                            "Stage {:03d} done after {:.4f} seconds".format(
                                opt_idx, elapsed
                            )
                        )

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    "Body fitting Orientation {} done after {:.4f} seconds".format(
                        or_idx, elapsed
                    )
                )
                tqdm.write(
                    "Body final loss val = {:.5f}".format(final_loss_val)
                )

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {
                "camera_" + str(key): val.detach().cpu().numpy()
                for key, val in camera.named_parameters()
            }
            result.update(
                {
                    key: val.detach().cpu().numpy()
                    for key, val in body_model.named_parameters()
                }
            )
            if use_vposer:
                result["pose_embedding"] = (
                    pose_embedding.detach().cpu().numpy()
                )
                body_pose = (
                    vposer.decode(pose_embedding, output_type="aa").reshape(
                        pose_embedding.shape[0], -1
                    )
                    if use_vposer
                    else None
                )
                result["body_pose"] = body_pose.detach().cpu().numpy()

        model_output = body_model(return_verts=True, body_pose=body_pose)
    return model_output, result
