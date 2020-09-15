import argparse
from pathlib import Path

import dominate
import dominate.tags as dtags
import pandas as pd

from epic import htmlgrid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smthgv2_root", type=str, default="/sequoia/data2/dataset/smthg-smthg-v2/"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="dataset split in [train|val|test]",
    )
    parser.add_argument(
        "--label_filter",
        type=str,
        default="Open",
        help="dataset split in [train|val|test]",
    )
    parser.add_argument(
        "--label_idxs",
        type=int,
        nargs="+",
        help="Keep only a subset of labels",
    )
    parser.add_argument(
        "--max_rows", default=100, type=int, help="path to image folders"
    )
    parser.add_argument("--destination", help="Path to html file", default="tmp.html")
    parser.add_argument(
        "--print", action="store_true", help="Prints html doc as string"
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle images")
    parser.add_argument("--max_img_size", help="Limit image size")
    args = parser.parse_args()

    doc = dominate.document("my doc")
    with doc.head:
        dtags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.css",
        )
        dtags.link(
            rel="stylesheet",
            href="https://www.rocq.inria.fr/cluster-willow/yhasson/markdown-reports/css/perso-git-markdown.css",
        )

    all_images = []

    if args.split == "val":
        split_suffix = "validation"
    else:
        split_suffix = args.split

    # Get all images for each folder
    smthg_root = Path(args.smthgv2_root)
    action_label_path = smthg_root / "something-something-v2-labels.csv"
    label_df = pd.read_csv(action_label_path, names=["action_label"], delimiter=";")
    action_labels = label_df.values[:, 0].tolist()
    if args.label_filter is not None:
        action_labels = [label for label in action_labels if args.label_filter in label]
        print(
            f"Kept {len(action_labels)} labels {action_labels} which contain '{args.label_filter}'"
        )
    if args.label_idxs is not None:
        action_labels = [action_labels[idx] for idx in args.label_idxs]
        print(
            f"Kept {len(action_labels)} labels {action_labels} at idxs {args.label_idxs}"
        )
    split_df = pd.read_csv(
        smthg_root / f"something-something-v2-{split_suffix}.csv",
        delimiter=";",
        names=["video_id", "action_label"],
    )
    df = split_df[split_df.action_label.isin(action_labels)]
    print(
        f"Kept {df.shape[0]} out of {split_df.shape[0]} videos with labels {action_labels}"
    )
    keep_rows = split_df.action_label.isin(action_labels)
    video_ids = df[keep_rows].video_id.values.tolist()

    # Arrange as list [{0: img_1_folder_0, 1:img_1_folder_1, ..}, ]
    grid = [
        [
            {
                "label": video_id,
                "type": "video",
                "path": smthg_root / "videos" / f"{video_id}.webm",
            }
        ]
        for video_id in video_ids
    ]

    with doc:
        with dtags.article(cls="markdown-body"):
            htmlgrid.html_grid(grid)

    if args.destination is not None:
        with open(args.destination, "w") as f:
            f.write(doc.render())
    print("Write html to {}".format(args.destination))
    if args.print:
        print(doc)
