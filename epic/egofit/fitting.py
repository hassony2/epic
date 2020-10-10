from collections import defaultdict
from pathlib import Path
import pickle

from tqdm import tqdm
from epic.smplifyx import optim_factory
from epic.egofit import egoviz


def fit_human(
    supervision,
    scene,
    iters=100,
    lr=0.1,
    optimizer="adam",
    save_root="tmp",
    optim_shape=False,
    viz_step=10,
):
    save_folder = Path(save_root) / f"opt{optimizer}_lr{lr:.4f}_it{iters:04d}"
    save_folder.mkdir(exist_ok=True, parents=True)
    optim_params = scene.get_optim_params()
    print(f"Optimizing {len(optim_params)} parameters")
    (body_optimizer, body_create_graph,) = optim_factory.create_optimizer(
        optim_params, optim_type=optimizer, lr=lr
    )

    losses = defaultdict(list)
    for iter_idx in tqdm(range(iters)):
        scene_outputs = scene.forward()

        body_optimizer.step()
        if iter_idx % viz_step == 0:
            egoviz.ego_viz(
                scene,
                scene_outputs,
                supervision,
                step_idx=iter_idx,
                save_folder=save_folder / "viz",
            )
    res = {"losses": losses}
    with (save_folder / "res.pkl").open("rb") as p_f:
        pickle.dump(res, p_f)
    return res
