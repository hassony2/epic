import os
from moviepy import editor as mpy
from matplotlib import pyplot as plt
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

from epic.io.tarutils import TarReader
from epic.viz import masksviz, handviz, hoaviz
from epic.hoa import gethoa, links
from epic.tracking import trackhoadf

from libyana.visutils import vizmp

focal = 150
CAMINTR = np.array([[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]])


def prepare_sequence(
    epic_root,
    video_id,
    start_frame,
    end_frame,
    obj_path,
    pickle_path="tmp.pkl",
    frame_step=2,
    fig=None,
    show_adv=False,
    split="train",
    no_tar=False,
    resize_factor=456 / 1920,  # from original to target resolution
    mask_extractor=None,
    hand_extractor=None,
    debug=False,
    camintr=CAMINTR,
    hoa_dets=None,
    hoa_root=None,
    track_padding=40,
    segm_padding=20,
    fps=60,
):
    if hoa_dets is None:
        hoa_dets = gethoa.load_video_hoa(video_id, hoa_root=hoa_root)
    frame_template = os.path.join(epic_root, "{}/{}/{}/frame_{:010d}.jpg")
    if not no_tar:
        tareader = TarReader()
    dump_list = []
    imgs = []
    # Track hands and objects
    dt = frame_step / fps
    hoa_dets = trackhoadf.track_hoa_df(
        hoa_dets,
        video_id=video_id,
        start_frame=max(1, start_frame - track_padding),
        end_frame=(min(end_frame + track_padding, hoa_dets.frame.max() - 1)),
        dt=dt,
    )
    # Add padding to segmented clip
    start_frame = max(1, start_frame - segm_padding)
    end_frame = min(end_frame + track_padding, hoa_dets.frame.max() - 1)
    for frame_idx in tqdm(range(start_frame, end_frame, frame_step)):
        if fig is None:
            fig = plt.figure()
        # frame_name = "frame_{frame_idx:010d}.jpg"
        # frame_subpath = f"./{frame_name}"
        fig.clf()
        if show_adv:
            ax = fig.add_subplot(2, 1, 2)
        else:
            ax = fig.add_subplot(1, 1, 1)
        img_path = frame_template.format(
            split, video_id[:3], video_id, frame_idx
        )
        if no_tar:
            img = Image.open(img_path)
            img = np.array(img)
        else:
            img = tareader.read_tar_frame(img_path)
            img = img[:, :, ::-1]
        ax.imshow(img)
        ax.set_title(f"{frame_idx:08d}")
        hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
        hoa_links = links.links_from_df(hoa_df, resize_factor=resize_factor)
        with torch.no_grad():
            # Extract object masks
            if mask_extractor is not None:
                res = mask_extractor.masks_from_df(
                    img, hoa_df, resize_factor=resize_factor
                )
            # Extract hands
            pred_hands = hand_extractor.hands_from_df(
                img, hoa_df, resize_factor=resize_factor
            )

            if len(pred_hands):
                # Draw hand renderings
                handviz.add_hands_viz(ax, img, pred_hands, camintr)

                # Draw hand boxes
                hoaviz.add_hoa_viz(
                    ax, hoa_df, resize_factor=resize_factor, debug=debug
                )
            if (mask_extractor is not None) and len(
                res["masks"]
            ):  # TODO remove
                labels = [
                    f"{cls}: {score:.2f}"
                    for cls, score in zip(res["classes"], res["scores"])
                ]
                masksviz.add_masks_viz(
                    ax, res["masks"], res["boxes"], labels=labels, debug=debug
                )
            img = vizmp.fig2np(fig)
            imgs.append(img)
            if debug:
                fig.savefig(f"tmp_{frame_idx:05d}.png")

            if pickle_path is not None:
                dump_list.append(
                    {
                        "masks": res["masks"],
                        "boxes": res["boxes"],
                        "obj_path": obj_path,
                        "hands": pred_hands,
                        "img_path": img_path,
                        "resize_factor": resize_factor,
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "links": hoa_links,
                    }
                )

    clip = mpy.ImageSequenceClip(imgs, fps=int(fps / frame_step / 2))
    # final_clip = mpy.clips_array([[clip,], [score_clip,]])
    clip.write_videofile(pickle_path.replace(".pkl", ".webm"))
    clip.write_videofile(pickle_path.replace(".pkl", ".mp4"))
    if pickle_path is not None:
        with open(pickle_path, "wb") as p_f:
            pickle.dump(dump_list, p_f)
        print(f"Saved info to {pickle_path}")
