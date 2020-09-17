def flip_coords(coords, crop_size=256, axis=0):
    coords[:, axis] = crop_size - coords[:, axis]
    return coords
