import os
from pathlib import Path
import shutil
from functools import partial


def path2img(path, local_folder="", call_nb=[0]):
    if local_folder:
        local_folder = Path(local_folder) / "imgs"
        local_folder.mkdir(exist_ok=True, parents=True)

        ext = path.split(".")[-1]
        img_name = f"{call_nb[0]:04d}.{ext}"
        dest_img_path = local_folder / img_name
        print(dest_img_path)
        shutil.copy(path, dest_img_path)
        rel_path = os.path.join("imgs", img_name)
    else:
        rel_path = path

    # Keep track of count number
    call_nb[0] += 1
    print(rel_path)
    return '<img src="' + str(rel_path) + '"/>'


def df2html(df, local_folder=""):
    """
    Convert df to html table, getting images for fields which contain 'img_path'
    in their name.
    """
    keys = list(df.keys())
    format_dicts = {}
    for key in keys:
        if "img_path" in key:
            format_dicts[key] = partial(path2img, local_folder=local_folder)

    df_html = df.to_html(escape=False, formatters=format_dicts)
    return df_html
