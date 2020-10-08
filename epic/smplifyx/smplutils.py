def prepare_params(smpl, batch_size):
    smpl_params = {}
    smpl_params["leye_pose"] = smpl.leye_pose.repeat(batch_size, 1)
    smpl_params["reye_pose"] = smpl.reye_pose.repeat(batch_size, 1)
    smpl_params["jaw_pose"] = smpl.jaw_pose.repeat(batch_size, 1)
    smpl_params["global_orient"] = smpl.global_orient.repeat(batch_size, 1)
    smpl_params["transl"] = smpl.transl.repeat(batch_size, 1)
    smpl_params["expression"] = smpl.expression.repeat(batch_size, 1)
    smpl_params["body_pose"] = smpl.body_pose.repeat(batch_size, 1)
    smpl_params["left_hand_pose"] = smpl.left_hand_pose.repeat(batch_size, 1)
    smpl_params["right_hand_pose"] = smpl.right_hand_pose.repeat(batch_size, 1)
    return smpl_params
