import itertools

KEY_MAPPING = {
    "loss_hand_v": "lhv",
    "lambda_hand_v": "lhvl",
    "loss_obj_mask": "lom",
    "lambda_obj_mask": "loml",
    "focal": "f",
    "optimizer": "opt",
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
            param_str += f"{key_str}={val:.2e}"
        else:
            param_str += f"{key_str}={val}"
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
        "lr",
        "loss_hand_v",
        "lambda_hand_v",
        "lambda_obj_mask",
        "loss_obj_mask",
        "focal",
        "optimizer",
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
