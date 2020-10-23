from motpy import Detection, MultiObjectTracker
import pandas as pd

from epic.hoa import gethoa
from epic.tracking import trackconv
from tqdm import tqdm


def print_track_info(tracks, track_type="object"):
    unique_tracks = tracks.track_id.unique()
    print(
        f"Got {len(unique_tracks)} {track_type} tracks with ids and lengths"
        f"{tracks.groupby('track_id').frame.nunique()}"
    )


def track_hoa_df(
    hoa_dets,
    dt=0.02,
    start_frame=0,
    end_frame=100,
    video_id=None,
    verbose=True,
    object_only=False,
    keep_longest=True,
):
    """
    Args:
        keep_longest (bool): find longest object track sequence
    """
    # Initialize track lists and tracker
    obj_tracker = MultiObjectTracker(dt=dt)
    tracked_obj = []

    if not object_only:
        lh_tracker = MultiObjectTracker(dt=dt)
        rh_tracker = MultiObjectTracker(dt=dt)

        # Intialize tracked dicts
        tracked_lh = []
        tracked_rh = []

    # Last non-empty df
    for frame_idx in tqdm(range(start_frame, end_frame)):
        hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
        obj_df = hoa_df[hoa_df.det_type == "object"]
        obj_dets = [
            Detection(gethoa.row2box(row)) for _, row in obj_df.iterrows()
        ]
        obj_tracker.step(detections=obj_dets)
        tracked_obj.extend(
            trackconv.track2dicts(
                obj_tracker.active_tracks(),
                frame_idx,
                video_id=video_id,
                det_type="object",
            )
        )
        if not object_only:
            lh_df = hoa_df[
                (hoa_df.det_type == "hand") & (hoa_df.side == "left")
            ]
            rh_df = hoa_df[
                (hoa_df.det_type == "hand") & (hoa_df.side == "right")
            ]
            lh_dets = [
                Detection(gethoa.row2box(row)) for _, row in lh_df.iterrows()
            ]
            rh_dets = [
                Detection(gethoa.row2box(row)) for _, row in rh_df.iterrows()
            ]
            lh_tracker.step(detections=lh_dets)
            rh_tracker.step(detections=rh_dets)
            tracked_lh.extend(
                trackconv.track2dicts(
                    lh_tracker.active_tracks(),
                    frame_idx,
                    video_id=video_id,
                    det_type="hand",
                    side="left",
                )
            )
            tracked_rh.extend(
                trackconv.track2dicts(
                    rh_tracker.active_tracks(),
                    frame_idx,
                    video_id=video_id,
                    det_type="hand",
                    side="right",
                )
            )
    if verbose:
        obj_tracks = pd.DataFrame(tracked_obj)
        if keep_longest:
            longest_track_idx = (
                obj_tracks.groupby("track_id").frame.nunique().idxmax()
            )
            # Filter object which has longest track
            tracked_obj = obj_tracks[obj_tracks.track_id == longest_track_idx]
        print_track_info(tracked_obj)
        if not object_only:
            lh_tracks = pd.DataFrame(tracked_lh)
            rh_tracks = pd.DataFrame(tracked_rh)
            print_track_info(lh_tracks, track_type="left hand")
            print_track_info(rh_tracks, track_type="right hand")
            tracked_hoa = pd.DataFrame(
                tracked_obj.to_dict("records") + tracked_lh + tracked_rh
            )
        else:
            tracked_hoa = pd.DataFrame(tracked_obj)
        if keep_longest:
            start_track_frame = tracked_obj.frame.min()
            end_track_frame = tracked_obj.frame.max()
            # Keep only region that focuses on longest track
            tracked_hoa = tracked_hoa[
                (tracked_hoa.frame >= start_track_frame)
                & (tracked_hoa.frame <= end_track_frame)
            ]
    return tracked_hoa
