import argparse
import os
from pathlib import Path

import dominate
import dominate.tags as dtags
import pandas as pd

from epic import htmlgrid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_names", type=str, nargs="+")
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

    grid_cells = []
    for folder_name in args.folder_names:
        folder_name = Path(folder_name)
        file_names = os.listdir(folder_name)
        folder_cells = []
        for file_name in file_names:
            file_path = folder_name / file_name
            cell = htmlgrid.auto_make_cell(file_path)
            folder_cells.append(cell)
        grid_cells.append(folder_cells)

    with doc:
        with dtags.article(cls="markdown-body"):
            htmlgrid.html_grid(grid_cells, transpose=True)

    if args.destination is not None:
        with open(args.destination, "w") as f:
            f.write(doc.render())
    print("Write html to {}".format(args.destination))
    if args.print:
        print(doc)
