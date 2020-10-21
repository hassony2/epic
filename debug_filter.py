from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
import pickle
from epic.egofit.scene import Scene
from epic.egofit import camera
from epic.egofit.preprocess import Preprocessor
from epic.egofit import exputils, fitargs
from epic.viz import lineviz

from libyana.visutils import vizmp
from libyana.exputils import argutils
from libyana.randomutils import setseeds
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("agg")

args = fitargs.fit_args()

if args.debug:
    torch.autograd.set_detect_anomaly(True)
    setseeds.set_all_seeds(0)

with open(args.pickle_path, "rb") as p_f:
    data = pickle.load(p_f)

render_size = (args.render_res, args.render_res)
preprocessor = Preprocessor(crop_size=render_size, debug=args.debug)
# Select frames uniformly for optimization
if args.frame_nb is not None:
    frame_idxs = np.linspace(0, len(data) - 1, args.frame_nb).astype(np.int)
    data = [data[idx] for idx in frame_idxs]

dt = 0.02
# Prepare supervision
if args.resume:
    with (Path(args.resume) / "res.pkl").open("rb") as p_f:
        resume_res = pickle.load(p_f)
    data_df = resume_res["data_df"]
    supervision = resume_res["supervision"]
else:
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
    scene.cuda()
    # Reload optimized state
    if args.resume:
        scene.load_state(Path(args.resume))
        print(
            f"Loaded scene from {args.resume}, resetting object translation."
        )
    scene.save_scene_clip(["tmp.webm", "tmp.mp4"], imgs=supervision["imgs"])
    # Initialize object by hand pose
    scene.reset_obj2hand()
    left_hand_pose = scene.egohuman.left_hand_pose
    right_hand_pose = scene.egohuman.right_hand_pose
    pose_embedding = scene.egohuman.pose_embedding

    # Smooth pose values
    smoothed_scene = deepcopy(scene)
    smoothed_scene.smooth_human_pose(order=arg_dict["rts_order"])

    smoothed_scene.save_scene_clip(
        ["tmp_smoothed.webm", "tmp_smoothed.mp4"], imgs=supervision["imgs"]
    )
    row_nb = 2
    col_nb = 3
    fig, axes = plt.subplots(row_nb, col_nb)
    ax = vizmp.get_axis(
        axes, row_idx=0, col_idx=0, row_nb=row_nb, col_nb=col_nb
    )
    lineviz.add_lines(ax, pose_embedding.transpose(1, 0))
    ax = vizmp.get_axis(
        axes, row_idx=0, col_idx=1, row_nb=row_nb, col_nb=col_nb
    )
    lineviz.add_lines(ax, left_hand_pose.transpose(1, 0))
    ax.set_title("left hand")
    ax = vizmp.get_axis(
        axes, row_idx=0, col_idx=2, row_nb=row_nb, col_nb=col_nb
    )
    lineviz.add_lines(ax, right_hand_pose.transpose(1, 0))
    ax.legend()
    ax.set_title("right hand")

    # Smooth pose_embedding
    ax = vizmp.get_axis(
        axes, row_idx=1, col_idx=0, row_nb=row_nb, col_nb=col_nb
    )
    lineviz.add_lines(
        ax,
        pose_embedding.transpose(1, 0),
        over_lines=smoothed_scene.egohuman.pose_embedding.transpose(1, 0),
    )

    # Smooth left hand
    ax = vizmp.get_axis(
        axes, row_idx=1, col_idx=1, row_nb=row_nb, col_nb=col_nb
    )
    lineviz.add_lines(
        ax,
        left_hand_pose.transpose(1, 0),
        over_lines=smoothed_scene.egohuman.left_hand_pose.transpose(1, 0),
    )
    ax.legend()

    # Smooth right hand
    ax = vizmp.get_axis(
        axes, row_idx=1, col_idx=2, row_nb=row_nb, col_nb=col_nb
    )
    lineviz.add_lines(
        ax,
        right_hand_pose.transpose(1, 0),
        over_lines=smoothed_scene.egohuman.right_hand_pose.transpose(1, 0),
    )
    ax.legend()

    fig.savefig(f"tmp_rts{arg_dict['rts_order']}_p10_q0_1_{dt}.png")
    # import pdb; pdb.set_trace()
