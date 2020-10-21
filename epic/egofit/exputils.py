import itertools

KEY_MAPPING = {
    "blend_gamma": "bg",
    "loss_hand_v": "lhv",
    "lambda_hand_v": "lhvl",
    "loss_link": "lnk",
    "lambda_link": "lnkl",
    "loss_obj_mask": "lom",
    "lambda_obj_mask": "loml",
    "lambda_obj_smooth": "losl",
    "lambda_body_smooth": "lbsl",
    "loss_smooth": "lsm",
    "mask_mode": "mm",
    "focal": "f",
    "optimizer": "opt",
    "rts_order": "rtso",
}


def get_arg_string(param_dict):
    param_str = ""
    for key in sorted(list(param_dict.keys())):
        val = param_dict[key]
        if key in KEY_MAPPING:
            key_str = KEY_MAPPING[key]
        else:
            key_str = key

        if isinstance(val, float):
            param_str += f"_{key_str}={val:.2e}"
        else:
            param_str += f"_{key_str}={val}"
    return param_str


def process_args(args):
    """
    Gets dictionary of arguments and list of representative experiment
    string (which uniquely identify the experimental run)

    Return:
        (list): [{param1_name: param1_val, ...}, ...]
        (list): [args1_str, args2_str, ...]
    """
    args = vars(args)
    multi_keys = [
        "blend_gamma",
        "lr",
        "loss_hand_v",
        "lambda_hand_v",
        "loss_link",
        "lambda_link",
        "lambda_obj_mask",
        "lambda_obj_smooth",
        "lambda_body_smooth",
        "loss_obj_mask",
        "loss_smooth",
        "mask_mode",
        "focal",
        "optimizer",
        "rts_order",
    ]
    multi_params = [args[f"{key}s"] for key in multi_keys]
    exp_params = itertools.product(*multi_params)
    param_dict_list = []
    param_strings = []
    for exp_param in exp_params:
        param_dict = {key: val for (key, val) in zip(multi_keys, exp_param)}
        param_dict_list.append(param_dict)
        param_string = get_arg_string(param_dict)
        param_strings.append(param_string)
    return param_dict_list, param_strings
