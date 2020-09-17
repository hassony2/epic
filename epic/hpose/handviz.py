from libyana.visutils import viz2d


def add_hand_viz(ax, hand_df, joint_idxs=False, score_thresh=0.2):
    for hand_idx, hand_det in hand_df.iterrows():
        joints2d = hand_det.joints2d
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
        scores = hand_det.joints2d_scores
        if scores.shape[0] > 21:
            kept_links = []
            for finger_idxs in links:
                score_finger_links = []
                for link_idx in finger_idxs:
                    if scores[link_idx] > score_thresh:
                        score_finger_links.append(link_idx)
                kept_links.append(score_finger_links)
            joint_labels = [f"{scores[idx]:.2f}" for idx in range(len(joints2d))]
        else:
            joint_labels = None
        viz2d.visualize_joints_2d(
            ax,
            joints2d,
            joint_labels=joint_labels,
            joint_idxs=joint_idxs,
            links=kept_links,
        )
