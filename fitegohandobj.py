import argparse
from pathlib import Path

import torch
import numpy as np
import pickle
from epic.egofit.scene import Scene
from epic.egofit import fitting
from epic.egofit import camera
from epic.egofit.preprocess import Preprocessor
from epic.egofit.egolosses import EgoLosses
from epic.egofit import exputils

from libyana.exputils import argutils
from libyana.randomutils import setseeds
import matplotlib

matplotlib.use("agg")

parser = argparse.ArgumentParser()
parser.add_argument("--pickle_path", default="tmp.pkl")
parser.add_argument("--optimizers", default=["adam"], nargs="+")
parser.add_argument("--mask_modes", default=["mask"], nargs="+")
parser.add_argument("--blend_gammas", default=[1e-2], type=float, nargs="+")
parser.add_argument("--lambda_hand_vs", default=[1], type=float, nargs="+")
parser.add_argument("--lambda_obj_masks", default=[1], type=float, nargs="+")
parser.add_argument("--lambda_links", default=[1], type=float, nargs="+")
parser.add_argument("--loss_links", default=["l1"], type=str, nargs="+")
parser.add_argument(
    "--loss_hand_vs", default=["l1"], type=str, nargs="+", choices=["l1", "l2"]
)
parser.add_argument(
    "--loss_obj_masks",
    default=["l1"],
    type=str,
    nargs="+",
    choices=["l1", "l2", "adapt"],
)
parser.add_argument("--focals", default=[150], type=float, nargs="+")
parser.add_argument("--viz_step", default=10, type=int)
parser.add_argument("--iters", default=400, type=int)
parser.add_argument("--lrs", default=[0.01], type=float, nargs="+")
parser.add_argument("--render_res", default=256, type=int, nargs="+")
parser.add_argument(
    "--loss_types",
    default=["adapt"],
    choices=["adapt", "l2", "l1", "adapt_dtf", "l2_dtf", "l1_dtf"],
    nargs="+",
)
parser.add_argument("--faces_per_pixels", default=[2], type=int, nargs="+")
parser.add_argument("--save_root", default="tmp")
parser.add_argument("--rot_nb", default=1, type=int)
parser.add_argument("--no_crop", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--resume", default="", type=str)

# Block parameters to ease optimization
parser.add_argument("--block_obj_scale", action="store_true")
parser.add_argument("--no_obj_optim", action="store_true")
parser.add_argument("--no_hand_optim", action="store_true")
parser.add_argument(
    "--frame_nb", default=2, type=int, help="Number of frames to optimize"
)
args = parser.parse_args()
argutils.print_args(args)
if args.no_hand_optim and args.no_obj_optim:
    raise ValueError(
        "--no_hand_optim and --no_obj_optim should not be both  set"
    )
if args.iters < args.viz_step:
    args.viz_step = args.iters - 1

if args.debug:
    torch.autograd.set_detect_anomaly(True)
    setseeds.set_all_seeds(0)


with open(args.pickle_path, "rb") as p_f:
    data = pickle.load(p_f)

render_size = (args.render_res, args.render_res)
preprocessor = Preprocessor(crop_size=render_size, debug=args.debug)
# Select frames uniformly for optimization
frame_idxs = np.linspace(0, len(data) - 1, args.frame_nb).astype(np.int)
data = [data[idx] for idx in frame_idxs]

# Prepare supervision
data_df = preprocessor.preprocess_df(data)
supervision = preprocessor.preprocess_supervision(data)

# Prepare camera
camrot = torch.eye(3)
camrot[0, 0] = -1
camrot[1, 1] = -1


img_size = supervision["imgs"][0].shape[:2]  # height, width
save_root = Path(args.save_root)
save_root.mkdir(exist_ok=True, parents=True)
argutils.save_args(args, save_root)
# Simulate multiple objects
args_list, args_str = exputils.process_args(args)
print(f"Running {len(args_list)} experiments !")
for run_idx, (arg_dict, arg_str) in enumerate(zip(args_list, args_str)):
    save_folder = save_root / arg_str
    save_folder.mkdir(exist_ok=True, parents=True)
    camintr = torch.Tensor(
        np.array(
            [
                [arg_dict["focal"], 0, 456 // 2],
                [0, arg_dict["focal"], 256 // 2],
                [0, 0, 1],
            ]
        )
    )
    cam = camera.PerspectiveCamera(
        camintr=camintr, rot=camrot, image_size=img_size
    )
    scene = Scene(
        data_df,
        cam,
        roi_bboxes=supervision["roi_bboxes"],
        render_size=render_size,
        blend_gamma=arg_dict["blend_gamma"],
    )
    # Reload optimized state
    if args.resume:
        scene.load_state(Path(args.resume))
        print(
            f"Loaded scene from {args.resume}, resetting object translation."
        )
    # Initialize object by hand pose
    scene.reset_obj2hand()
    egolosses = EgoLosses(
        lambda_hand_v=arg_dict["lambda_hand_v"],
        loss_hand_v=arg_dict["loss_hand_v"],
        lambda_link=arg_dict["lambda_link"],
        loss_link=arg_dict["loss_link"],
        lambda_obj_mask=arg_dict["lambda_obj_mask"],
        loss_obj_mask=arg_dict["loss_obj_mask"],
        mask_mode=arg_dict["mask_mode"],
    )

    res = fitting.fit_human(
        data,
        supervision,
        scene,
        egolosses,
        iters=args.iters,
        lr=arg_dict["lr"],
        optimizer=arg_dict["optimizer"],
        save_folder=save_folder,
        viz_step=args.viz_step,
        block_obj_scale=args.block_obj_scale,
        no_obj_optim=args.no_obj_optim,
        no_hand_optim=args.no_hand_optim,
    )
    res["opts"] = arg_dict
    res["args"] = vars(args)
    scene.save_state(save_folder)
    with (save_folder / "res.pkl").open("wb") as p_f:
        pickle.dump(res, p_f)
    print(f"Iteration {run_idx} done !")

print("All iterations done !")
