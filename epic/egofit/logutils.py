import os
from pathlib import Path
import shutil
from functools import partial
import pandas as pd


def drop_redundant_columns(df):
    """
    If dataframe contains multiple lines, drop the ones for which the column
    contains equal values
    """
    if len(df) > 1:
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        print(f"Dropping {list(cols_to_drop)}")
        df = df.drop(cols_to_drop, axis=1)
    return df


def make_collapsible(html_str, collapsible_idx=0):
    """
    Create collapsible button to selectively hide large html items such as images
    """
    pref = (
        f'<button data-toggle="collapse" data-target="#demo{collapsible_idx}">'
        "Toggle show image</button>"
        f'<div id="demo{collapsible_idx}" class="collapse">'
    )
    suf = "</div>"
    return pref + html_str + suf


def path2img(path, local_folder="", collabsible=True, call_nb=[0]):
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
    img_str = '<img src="' + str(rel_path) + '"/>'
    if collabsible:
        img_str = make_collapsible(img_str, call_nb[0])
    return img_str


def df2html(df, local_folder="", drop_redundant=True):
    """
    Convert df to html table, getting images for fields which contain 'img_path'
    in their name.
    """
    keys = list(df.keys())
    format_dicts = {}
    for key in keys:
        if "img_path" in key:
            format_dicts[key] = partial(path2img, local_folder=local_folder)

    if drop_redundant:
        df = drop_redundant_columns(df)

    df_html = df.to_html(escape=False, formatters=format_dicts)
    return df_html
