import argparse
import itertools
from pathlib import Path

import torch
import numpy as np
import pickle
from epic.egofit.scene import Scene
from epic.egofit import fitting
from epic.egofit import camera
from epic.egofit.preprocess import Preprocessor
from epic.egofit.egolosses import EgoLosses

from libyana.exputils import argutils
from libyana.randomutils import setseeds

parser = argparse.ArgumentParser()
parser.add_argument("--pickle_path", default="tmp.pkl")
parser.add_argument("--radius", default=0.1, type=float)
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--lambda_hand_vs", default=[1], type=float, nargs="+")
parser.add_argument("--lambda_obj_masks", default=[1], type=float, nargs="+")
parser.add_argument(
    "--loss_hand_vs", default=["l1"], type=str, nargs="+", choices=["l1", "l2"]
)
parser.add_argument(
    "--loss_obj_masks",
    default=["l1"],
    type=str,
    nargs="+",
    choices=["l1", "l2"],
)
parser.add_argument("--focal", default=200, type=float)
parser.add_argument("--z_offs", default=[0.3], type=float, nargs="+")
parser.add_argument("--viz_step", default=10, type=int)
parser.add_argument("--iters", default=100, type=int)
parser.add_argument("--lrs", default=[0.01], type=float, nargs="+")
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
parser.add_argument(
    "--frame_nb", default=2, type=int, help="Number of frames to optimize"
)
args = parser.parse_args()
argutils.print_args(args)
if args.debug:
    torch.autograd.set_detect_anomaly(True)
    setseeds.set_all_seeds(0)


with open(args.pickle_path, "rb") as p_f:
    data = pickle.load(p_f)

preprocessor = Preprocessor()
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

camintr = torch.Tensor(
    np.array([[args.focal, 0, 456 // 2], [0, args.focal, 256 // 2], [0, 0, 1]])
)

img_size = supervision["imgs"][0].shape[:2]  # height, width

# Simulate multiple objects
for lr, lohv, lhv, loom, lom in itertools.product(
    args.lrs,
    args.loss_hand_vs,
    args.lambda_hand_vs,
    args.loss_obj_masks,
    args.lambda_obj_masks,
):
    save_folder = Path(args.save_root) / (
        f"opt{args.optimizer}_lr{lr:.4f}_it{args.iters:04d}"
        f"lhv{lohv}{lhv:.2e}_lom{loom}{lom:.2e}"
    )
    save_folder.mkdir(exist_ok=True, parents=True)
    argutils.save_args(args, save_folder)
    cam = camera.PerspectiveCamera(
        camintr=camintr,
        rot=camrot,
        image_size=img_size,
    )
    scene = Scene(data_df, cam)
    egolosses = EgoLosses(
        lambda_hand_v=lhv,
        loss_hand_v=lohv,
        lambda_obj_mask=lom,
        loss_obj_mask=loom,
    )

    res = fitting.fit_human(
        data,
        supervision,
        scene,
        egolosses,
        iters=args.iters,
        lr=lr,
        optimizer=args.optimizer,
        save_folder=save_folder,
        viz_step=args.viz_step,
    )
    res["opts"] = vars(args)
    with (save_folder / "res.pkl").open("wb") as p_f:
        pickle.dump(res, p_f)
