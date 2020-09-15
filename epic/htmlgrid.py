import dominate.tags as dtags


def auto_make_cell(path, label=None):
    if str(path).split(".")[-1].lower() in ["mp4", "webm"]:
        cell = {
            "label": label,
            "type": "video",
            "path": path,
        }
    return cell


def html_grid(grid, transpose=False):
    if transpose:
        grid = list(map(list, zip(*grid)))
    with dtags.table().add(dtags.tbody()):
        for row_idx, row in enumerate(grid):
            with dtags.tr():
                for cell_idx, cell in enumerate(row):
                    cell_type = cell["type"]
                    if cell_type == "txt":
                        if "text" not in cell:
                            raise ValueError(
                                "Expected grid cell of type 'txt'"
                                " to have field 'text'"
                            )
                        dtags.td().add(dtags.p(cell["text"]))
                    elif cell_type == "video":
                        if "path" not in cell:
                            raise ValueError(
                                "Expected grid cell of type 'video'"
                                " to have field 'path'"
                            )
                        video_path = cell["path"]
                        if str(video_path).lower().endswith("webm"):
                            vid_type = "video/webm"
                        if str(video_path).lower().endswith("mp4"):
                            vid_type = "video/mp4"
                        with dtags.td():
                            dtags.video(controls=True, autoplay=False).add(
                                dtags.source(src=video_path, type=vid_type)
                            )
