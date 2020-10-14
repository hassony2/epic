from epic import boxutils


def links_from_df(hoa_df, resize_factor=1):
    link_df = hoa_df[hoa_df.hoa_link == hoa_df.hoa_link]
    links = []
    for link_idx, link_row in link_df.iterrows():
        box_ltrb = boxutils.dfbox_to_norm(
            link_row, resize_factor=resize_factor
        )
        box_center = [
            box_ltrb[2] + box_ltrb[0],
            box_ltrb[3] + box_ltrb[1],
        ]
        link = {
            "type": link_row.hoa_link,
            "link_source": box_center,
            "obj_offset": [link_row.obj_offx, link_row.obj_offy],
            "side": link_row.side,
        }
        links.append(link)
    return links
