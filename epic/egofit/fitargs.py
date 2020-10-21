import argparse

from libyana.exputils import argutils


def fit_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", default="tmp.pkl")
    parser.add_argument(
        "--frame_nb", type=int, help="Number of frames to optimize"
    )

    # Optimizer pars
    parser.add_argument("--optimizers", default=["adam"], nargs="+")
    parser.add_argument("--iters", default=400, type=int)
    parser.add_argument("--lrs", default=[0.01], type=float, nargs="+")

    # RTS smoothing params
    parser.add_argument("--rts_orders", default=[1], type=int, nargs="+")
    parser.add_argument("--rts_dts", default=[1], type=float, nargs="+")

    # Loss params
    parser.add_argument("--mask_modes", default=["mask"], nargs="+")
    parser.add_argument(
        "--blend_gammas", default=[100000], type=float, nargs="+"
    )
    parser.add_argument("--lambda_hand_vs", default=[1], type=float, nargs="+")
    parser.add_argument(
        "--lambda_obj_masks", default=[1], type=float, nargs="+"
    )
    parser.add_argument(
        "--lambda_obj_smooths", default=[0], type=float, nargs="+"
    )
    parser.add_argument(
        "--lambda_body_smooths", default=[0], type=float, nargs="+"
    )
    parser.add_argument(
        "--loss_smooths",
        default=["l1"],
        type=str,
        nargs="+",
        choices=["l1", "l2", "adapt"],
    )
    parser.add_argument("--lambda_links", default=[1], type=float, nargs="+")
    parser.add_argument("--loss_links", default=["l1"], type=str, nargs="+")
    parser.add_argument(
        "--loss_hand_vs",
        default=["l1"],
        type=str,
        nargs="+",
        choices=["l1", "l2"],
    )
    parser.add_argument(
        "--loss_obj_masks",
        default=["l1"],
        type=str,
        nargs="+",
        choices=["l1", "l2", "adapt"],
    )

    # Block parameters to ease optimization
    parser.add_argument("--block_obj_scale", action="store_true")
    parser.add_argument("--no_obj_optim", action="store_true")
    parser.add_argument("--no_hand_optim", action="store_true")
    parser.add_argument("--focals", default=[150], type=float, nargs="+")
    parser.add_argument("--render_res", default=256, type=int, nargs="+")
    parser.add_argument("--faces_per_pixels", default=[2], type=int, nargs="+")

    parser.add_argument("--viz_step", default=10, type=int)
    parser.add_argument("--rot_nb", default=1, type=int)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", default="", type=str)

    # Experiment parameters
    parser.add_argument("--save_root", default="tmp")

    args = parser.parse_args()
    argutils.print_args(args)
    if args.no_hand_optim and args.no_obj_optim:
        raise ValueError(
            "--no_hand_optim and --no_obj_optim should not be both  set"
        )
    if args.iters < args.viz_step:
        args.viz_step = args.iters - 1
    return args
