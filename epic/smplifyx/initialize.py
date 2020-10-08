from collections import namedtuple

import numpy as np
import torch

from epic.smplifyx.utils import JointMapper, smpl_to_openpose

JOINTS_IGN = list(range(4)) + list(range(5, 7)) + list(range(8, 25))
JOINTS_LH = [4] + list(range(25, 46))
JOINTS_RH = [7] + list(range(46, 67))

NUM_BODY_JOINTS = 25
NUM_HAND_JOINTS = 20

Keypoints = namedtuple("Keypoints", ["keypoints", "gender_gt", "gender_pd"])


def get_model_params(args, dtype=torch.float64):
    model2data = smpl_to_openpose(
        args.get("model_type"),
        use_hands=args.get("use_hands"),
        use_face=args.get("use_face"),
        use_face_contour=args.get("use_face_contour"),
        openpose_format="coco25",
    )
    joint_mapper = JointMapper(model2data)
    model_params = dict(
        model_path=args.get("model_folder"),
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not args.get("use_vposer"),
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=dtype,
        **args,
    )
    return model_params


def get_joint_weights(use_hands=True, hand_sides=["right", "left"]):
    num_joints = NUM_BODY_JOINTS + NUM_HAND_JOINTS * use_hands * 2
    # The weights for the joint terms in the optimization
    optim_weights = np.ones((num_joints + 2 * use_hands), dtype=np.float32)

    # Neck, Left and right hip
    # These joints are ignored because SMPL has no neck joint and the
    # annotation of the hips is ambiguous.
    joints_to_ign = JOINTS_IGN
    if "right" not in hand_sides:
        joints_to_ign = joints_to_ign + JOINTS_RH
    if "left" not in hand_sides:
        joints_to_ign = joints_to_ign + JOINTS_LH
    if joints_to_ign is not None and -1 not in joints_to_ign:
        optim_weights[joints_to_ign] = 0.0
    return torch.tensor(optim_weights).float()


def get_keypoints(hand_left=None, hand_right=None):
    body_keypoints = np.zeros((NUM_BODY_JOINTS, 3))
    if hand_left is not None:
        hand_left = np.concatenate(
            [hand_left, np.ones((hand_left.shape[0], 1))], axis=1
        )
    if hand_right is not None:
        hand_right = np.concatenate(
            [hand_right, np.ones((hand_right.shape[0], 1))], axis=1
        )
    keypoints = np.concatenate([body_keypoints, hand_left, hand_right], axis=0)
    return Keypoints(keypoints=keypoints, gender_gt=[], gender_pd=[])
